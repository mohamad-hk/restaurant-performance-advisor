from fastapi import FastAPI, Request
import pandas as pd
import joblib

app = FastAPI() 

saved = joblib.load("model.pkl")
model = saved["model"] if isinstance(saved, dict) else saved

df = pd.read_csv("weekly_data.csv")
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

    last_row = subset.sort_values(["year", "week"]).iloc[-1]
    X = last_row[features].to_frame().T
    X = X.astype(float)

    pred = model.predict(X)[0]
    return {
        "code": code,
        "predicted_next_rate": float(pred)
    }
