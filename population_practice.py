import pandas as pd
import numpy as np

# === Load data ===
df = pd.read_csv("population_practice.csv")

print("\n=== Preview of your data ===")
print(df.head())

# === Fit regression for each subregion ===
coefs = {}
for subrg, g in df.groupby("subregion"):
    g = g.sort_values("year")
    a, b = np.polyfit(g["year"].values, g["population"].values, deg=1)
    coefs[subrg] = {"a": a, "b": b}

print("\n=== Regression equations ===")
for subrg, v in coefs.items():
    print(f"{subrg:15s}: population = {v['a']:.2f} * year + {v['b']:.2f}")

# === Predict for future years ===
future_years = list(range(2030, 2050))
rows = []
for subrg, v in coefs.items():
    for year in future_years:
        pop = v['a'] * year + v['b']
        rows.append({"subregion": subrg, "year": year, "predicted_population": round(pop, 0)})

pred_df = pd.DataFrame(rows)
pred_df.to_csv("predicted_population.csv", index=False)

print("\nPredictions saved to predicted_population.csv")

