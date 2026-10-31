import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

d = pd.read_csv("uniswap_v3_swaps_synthetic.csv")
d["dir"] = np.sign(d.amount0)
d["self_all"] = ((d.sender == d.recipient) & (d.recipient == d.origin)).astype(int)

same_tx = d.groupby(["origin", "transaction_hash"]).dir.nunique().gt(1).groupby("origin").any()
p = d.groupby(["origin", "pool_address"]).agg(n=("dir", "size"), nd=("dir", "nunique"), net=("amount0", "sum"),
                                              gross=("amount0", lambda s: s.abs().sum()))
round_trip = ((p.n >= 2) & (p.nd > 1) & (p.net.abs() / p.gross < 0.01)).groupby("origin").any()
self_all = d.groupby("origin").self_all.max().astype(bool)
y = (same_tx | round_trip | self_all).astype(int)

X = d.groupby("origin").agg(
    n_swaps=("swap_id", "size"), mean_usd=("amountUSD", "mean"), std_usd=("amountUSD", "std"),
    max_usd=("amountUSD", "max"), n_pools=("pool_address", "nunique"), mean_gas=("gas_used", "mean"),
    gas_price=("gas_price", "mean"), fee_tier=("fee_tier", "mean"), n_blocks=("block_number", "nunique"),
    buy_share=("dir", lambda s: (s > 0).mean()), t_min=("timestamp", "min"), t_max=("timestamp", "max")).fillna(0)
X["span_days"] = (X.t_max - X.t_min) / 86400
X["swaps_per_block"] = X.n_swaps / X.n_blocks
X["usd_cv"] = (X.std_usd / X.mean_usd).fillna(0)
X["log_gas_price"] = np.log1p(X.gas_price)
X = X.drop(columns=["t_min", "t_max", "gas_price"])
y = y.loc[X.index]
print("wallets:", len(X), "| flagged:", y.sum(), f"({y.mean():.1%})")

cv = StratifiedKFold(5, shuffle=True, random_state=42)
metrics = ["f1", "roc_auc", "average_precision"]


def evaluate(model, data):
    r = cross_validate(model, data, y, cv=cv, scoring=metrics)
    return {m: r[f"test_{m}"].mean() for m in metrics}


def base():
    return [("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("dt", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
            ("rf", RandomForestClassifier(200, class_weight="balanced", random_state=42)),
            ("knn", KNeighborsClassifier(7))]


def scaled(m):
    return Pipeline([("scale", StandardScaler()), ("model", m)])


metas = {"LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
         "DecisionTree": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
         "RandomForest": RandomForestClassifier(100, class_weight="balanced", random_state=42),
         "GradientBoosting": GradientBoostingClassifier(random_state=42)}

rows = {name: evaluate(scaled(m), X) for name, m in base()}
for name, meta in metas.items():
    stack = StackingClassifier(base(), final_estimator=meta, cv=5, stack_method="predict_proba")
    rows[f"Stack[{name}]"] = evaluate(scaled(stack), X)
a1 = pd.DataFrame(rows).T.round(3)
print("\nA1: base models vs stacking\n", a1)

corr = X.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
upper = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
print("\nA2: strongest correlations\n", upper[upper.abs() > 0.8].sort_values(key=abs, ascending=False).round(2))

Z = StandardScaler().fit_transform(X)
cum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
k = int(np.argmax(cum >= 0.99) + 1)
print(f"\nA3: {k} of {X.shape[1]} components keep 99% variance")
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.99, color="r", ls="--")
plt.xlabel("Components")
plt.ylabel("Cumulative explained variance")
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150)


def pca_pipe(m):
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(0.99)), ("model", m)])


pca_rows = {name: evaluate(pca_pipe(m), X) for name, m in base()}
for name, meta in metas.items():
    stack = StackingClassifier(base(), final_estimator=meta, cv=5, stack_method="predict_proba")
    pca_rows[f"Stack[{name}]"] = evaluate(pca_pipe(stack), X)
a3 = pd.DataFrame(pca_rows).T.round(3)
print("\nA3: PCA features\n", a3)
print("\nChange (PCA - original)\n", (a3 - a1).round(3))
