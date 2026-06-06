"""
predict.py — Quick CLI predictor for the California Housing model
Usage: python predict.py
"""
import pickle, numpy as np

with open('model.pkl', 'rb') as f:
    obj = pickle.load(f)
model, scaler, features = obj['model'], obj['scaler'], obj['features']

print("\n🏠 California House Price Predictor")
print("=" * 45)
print("Enter block-group info (press Enter for default):\n")

defaults = [5.0, 25.0, 5.5, 1.05, 1200, 2.8, 37.5, -122.0]
vals = []
for feat, default in zip(features, defaults):
    raw = input(f"  {feat:<14} [{default}]: ").strip()
    vals.append(float(raw) if raw else default)

X = scaler.transform([vals])
pred = model.predict(X)[0]

print(f"\n{'=' * 45}")
print(f"  💰 Predicted Median House Value: ${pred*100:.1f}k")
print(f"{'=' * 45}\n")
