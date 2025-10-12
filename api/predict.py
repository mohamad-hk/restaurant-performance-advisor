import os
from fastapi import FastAPI, Request
import pandas as pd
import joblib

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")
data_path = os.path.join(BASE_DIR, "weekly_data.csv")

saved = joblib.load(model_path)
model = saved["model"] if isinstance(saved, dict) else saved

df = pd.read_csv(data_path)
df = df.sort_values(["code", "year", "week"]).reset_index(drop=True)

features = [
    "mean_rate", "rate_lag1", "rate_lag2", "rate_lag3", "rate_lag4",
    "delta_rate", "delta2_rate", "acceleration_rate",
    "rolling_mean_2w", "rolling_mean_3w", "rolling_mean_4w"
]

@app.post("/")
async def predict(request: Request):  
    data = await request.json()
    code = data.get("code")
    subset = df[df["code"] == code]
    if subset.empty:
        return {"error": f"Code {code} not found in dataset"}

    last_row = subset.sort_values(["year", "week"]).iloc[-1]
    X = last_row[features].to_frame().T.astype(float)
    pred = model.predict(X)[0]

    return {
        "code": code,
        "predicted_next_rate": float(pred),
        "last_mean_rate": float(last_row["mean_rate"])
    }

