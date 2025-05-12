import torch
import pandas as pd
import numpy as np

MODEL_PATH = "models/end_to_end.pt"
CSV_PATH = "data/401_labeled_output.csv"
WINDOW_SIZE = 100
FEATURE_COLS = ['channel', 'rank', 'bankgroup', 'bank', 'row', 'column']

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=FEATURE_COLS)
df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)

model = torch.jit.load(MODEL_PATH)
model.eval()

open_count = 0
closed_count = 0

for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE][FEATURE_COLS].values
    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(window_tensor)
        pred = output.argmax(dim=1).item()

        print(f"Model output: {output[0].tolist()}, Prediction: {pred}")

        if pred == 0:
            print("Close Prob: 1, Open Prob: 0")
            closed_count += 1
        else:
            print("Close Prob: 0, Open Prob: 1")
            open_count += 1

print("\nSummary:")
print(f"Total windows: {open_count + closed_count}")
print(f"Predicted open:  {open_count}")
print(f"Predicted closed: {closed_count}")