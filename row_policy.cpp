#include <vector>

#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/scheduler.h"
#include "dram_controller/rowpolicy.h"
#include "torch/script.h"

namespace Ramulator {

class OpenRowPolicy : public IRowPolicy, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, OpenRowPolicy, "OpenRowPolicy", "Open Row Policy.")

public:
    void init() override { }

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    void update(bool request_found, ReqBuffer::iterator& req_it) override {
        // OpenRowPolicy does not need to take any actions
    }
};

class ClosedRowPolicy : public IRowPolicy, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, ClosedRowPolicy, "ClosedRowPolicy", "Close Row Policy.")

private:
    IDRAM* m_dram;
    int m_PRE_req_id = -1;
    int m_cap = -1;
    int m_rank_level = -1;
    int m_bankgroup_level = -1;
    int m_bank_level = -1;
    int m_row_level = -1;
    int m_num_ranks = -1;
    int m_num_bankgroups = -1;
    int m_num_banks = -1;

    int s_num_close_reqs = 0;

    std::vector<uint64_t> m_col_accesses;

public:
    void init() override { }

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
        m_ctrl = cast_parent<IDRAMController>();
        m_dram = m_ctrl->m_dram;

        m_cap = param<int>("cap").default_val(10000000); // TODO

        m_rank_level = m_dram->m_levels("rank");
        m_bankgroup_level = m_dram->m_levels("bankgroup");
        m_bank_level = m_dram->m_levels("bank");
        m_row_level = m_dram->m_levels("row");

        m_PRE_req_id = m_dram->m_requests("close-row");

        m_num_ranks = m_dram->get_level_size("rank");
        m_num_bankgroups = m_dram->get_level_size("bankgroup");
        m_num_banks = m_dram->get_level_size("bank");
        m_col_accesses.resize(m_num_banks * m_num_bankgroups * m_num_ranks, 0);

        register_stat(s_num_close_reqs).name("num_close_reqs");
    }

    void update(bool request_found, ReqBuffer::iterator& req_it) override {
        if (!request_found) return;

        if (m_dram->m_command_meta(req_it->command).is_closing ||
            m_dram->m_command_meta(req_it->command).is_refreshing)  // PRE or REF
        {
            if (req_it->addr_vec[m_bankgroup_level] == -1 &&
                req_it->addr_vec[m_bank_level] == -1) // all bank closes
            {
                for (int b = 0; b < m_num_banks; b++) {
                    for (int bg = 0; bg < m_num_bankgroups; bg++) {
                        int rank_id = req_it->addr_vec[m_rank_level];
                        int flat_bank_id = b + bg * m_num_banks + rank_id * m_num_banks * m_num_bankgroups;
                        m_col_accesses[flat_bank_id] = 0;
                    }
                }
            } else if (req_it->addr_vec[m_bankgroup_level] == -1) { // same bank closes
                for (int bg = 0; bg < m_num_bankgroups; bg++) {
                    int bank_id = req_it->addr_vec[m_bank_level];
                    int rank_id = req_it->addr_vec[m_rank_level];
                    int flat_bank_id = bank_id + bg * m_num_banks + rank_id * m_num_banks * m_num_bankgroups;
                    m_col_accesses[flat_bank_id] = 0;
                }
            } else { // single bank closes (PRE, VRR, RDA, WRA)
                int flat_bank_id = req_it->addr_vec[m_bank_level] +
                                   req_it->addr_vec[m_bankgroup_level] * m_num_banks +
                                   req_it->addr_vec[m_rank_level] * m_num_banks * m_num_bankgroups;

                m_col_accesses[flat_bank_id] = 0;
            }
        } else if (m_dram->m_command_meta(req_it->command).is_accessing) // RD or WR
        {
            int flat_bank_id = req_it->addr_vec[m_bank_level] +
                               req_it->addr_vec[m_bankgroup_level] * m_num_banks +
                               req_it->addr_vec[m_rank_level] * m_num_banks * m_num_bankgroups;

            m_col_accesses[flat_bank_id]++;

            if (m_col_accesses[flat_bank_id] >= m_cap) {
                Request req(req_it->addr_vec, m_PRE_req_id);
                m_ctrl->priority_send(req);
                m_col_accesses[flat_bank_id] = 0;
                s_num_close_reqs++;
            }
        }
    }
};

class HybridRowPolicy : public IRowPolicy, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, HybridRowPolicy, "HybridRowPolicy", "Hybrid Row Policy.")

private:
    IDRAM* m_dram;
    int m_PRE_req_id = -1;
    int m_cap = -1;
    int m_rank_level = -1;
    int m_bankgroup_level = -1;
    int m_bank_level = -1;
    int m_row_level = -1;
    int m_num_ranks = -1;
    int m_num_bankgroups = -1;
    int m_num_banks = -1;

    int s_num_close_reqs = 0;
    const int window_size = 100;
    const int feature_dim = 6;

    std::vector<uint64_t> m_col_accesses;
    std::vector<std::array<float, 6>> instruction_window;

    torch::jit::script::Module m_model;

    std::array<float, 6> extract_features(const ReqBuffer::iterator& req_it) {
        return {
            static_cast<float>(req_it->addr_vec[m_dram->m_levels("channel")]),
            static_cast<float>(req_it->addr_vec[m_rank_level]),
            static_cast<float>(req_it->addr_vec[m_bankgroup_level]),
            static_cast<float>(req_it->addr_vec[m_bank_level]),
            static_cast<float>(req_it->addr_vec[m_row_level]),
            static_cast<float>(req_it->addr_vec[m_dram->m_levels("column")])
        };
    }

public:
    void init() override { }

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
        m_ctrl = cast_parent<IDRAMController>();
        m_dram = m_ctrl->m_dram;

        m_cap = param<int>("cap").default_val(10000000); // fallback

        m_rank_level = m_dram->m_levels("rank");
        m_bankgroup_level = m_dram->m_levels("bankgroup");
        m_bank_level = m_dram->m_levels("bank");
        m_row_level = m_dram->m_levels("row");

        m_PRE_req_id = m_dram->m_requests("close-row");

        m_num_ranks = m_dram->get_level_size("rank");
        m_num_bankgroups = m_dram->get_level_size("bankgroup");
        m_num_banks = m_dram->get_level_size("bank");
        m_col_accesses.resize(m_num_banks * m_num_bankgroups * m_num_ranks, 0);

        register_stat(s_num_close_reqs).name("num_close_reqs");

        m_model = torch::jit::load("models/end_to_end.pt");
    }

    void update(bool request_found, ReqBuffer::iterator& req_it) override {
        if (!request_found) return;

        if (m_dram->m_command_meta(req_it->command).is_closing ||
            m_dram->m_command_meta(req_it->command).is_refreshing) // PRE or REF
        {
            if (req_it->addr_vec[m_bankgroup_level] == -1 &&
                req_it->addr_vec[m_bank_level] == -1) // all bank closes
            {
                for (int b = 0; b < m_num_banks; b++) {
                    for (int bg = 0; bg < m_num_bankgroups; bg++) {
                        int rank_id = req_it->addr_vec[m_rank_level];
                        int flat_bank_id = b + bg * m_num_banks + rank_id * m_num_banks * m_num_bankgroups;
                        m_col_accesses[flat_bank_id] = 0;
                    }
                }
            } else if (req_it->addr_vec[m_bankgroup_level] == -1) { // same bank closes
                for (int bg = 0; bg < m_num_bankgroups; bg++) {
                    int bank_id = req_it->addr_vec[m_bank_level];
                    int rank_id = req_it->addr_vec[m_rank_level];
                    int flat_bank_id = bank_id + bg * m_num_banks + rank_id * m_num_banks * m_num_bankgroups;
                    m_col_accesses[flat_bank_id] = 0;
                }
            } else { // single bank closes
                int flat_bank_id = req_it->addr_vec[m_bank_level] +
                                   req_it->addr_vec[m_bankgroup_level] * m_num_banks +
                                   req_it->addr_vec[m_rank_level] * m_num_banks * m_num_bankgroups;

                m_col_accesses[flat_bank_id] = 0;
            }
        } else if (m_dram->m_command_meta(req_it->command).is_accessing) // RD or WR
        {
            int flat_bank_id = req_it->addr_vec[m_bank_level] +
                               req_it->addr_vec[m_bankgroup_level] * m_num_banks +
                               req_it->addr_vec[m_rank_level] * m_num_banks * m_num_bankgroups;

            m_col_accesses[flat_bank_id]++;

            instruction_window.push_back(extract_features(req_it));

            if (instruction_window.size() == window_size) {
                std::vector<float> input_vec;
                for (const auto& instr : instruction_window)
                    input_vec.insert(input_vec.end(), instr.begin(), instr.end());

                torch::Tensor input = torch::from_blob(input_vec.data(), {1, window_size, feature_dim}, torch::kFloat32).clone();
                torch::Tensor output = m_model.forward({input}).toTensor();
                int prediction = output.argmax(1).item<int>();

                if (prediction == 0) {
                    for (int rank = 0; rank < m_num_ranks; ++rank) {
                        for (int bg = 0; bg < m_num_bankgroups; ++bg) {
                            for (int bank = 0; bank < m_num_banks; ++bank) {
                                std::vector<int> addr_vec = req_it->addr_vec;
                                addr_vec[m_rank_level] = rank;
                                addr_vec[m_bankgroup_level] = bg;
                                addr_vec[m_bank_level] = bank;
                
                                Request precharge_req(addr_vec, m_PRE_req_id);
                                m_ctrl->priority_send(precharge_req);
                
                                int flat_bank_id = bank + bg * m_num_banks + rank * m_num_banks * m_num_bankgroups;
                                m_col_accesses[flat_bank_id] = 0;
                            }
                        }
                    }
                    s_num_close_reqs++;
                }                

                instruction_window.clear();
            }
        }
    }
};

} // namespace Ramulator
