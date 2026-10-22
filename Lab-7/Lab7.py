import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


def entropy(y):
    p = np.unique(y, return_counts=True)[1] / len(y)
    return -np.sum(p * np.log2(p))


def gini(y):
    p = np.unique(y, return_counts=True)[1] / len(y)
    return 1 - np.sum(p ** 2)


def bin_feature(v, n_bins=4, method='width'):
    v = np.asarray(v, float)
    if method == 'width':
        edges = np.linspace(v.min(), v.max(), n_bins + 1)[1:-1]
    else:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges = edges[edges > v.min()]
    return np.digitize(v, edges), edges


def impurity(c, crit='entropy'):
    p = c / c.sum(-1, keepdims=True)
    if crit == 'gini':
        return 1 - (p ** 2).sum(-1)
    return -(p * np.log2(np.where(p > 0, p, 1))).sum(-1)


def info_gain(codes, y, crit='entropy'):
    k = y.max() + 1
    t = np.bincount(codes * k + y, minlength=(codes.max() + 1) * k).reshape(-1, k)
    t = t[t.sum(1) > 0]
    return impurity(t.sum(0), crit) - (t.sum(1) / len(y) * impurity(t, crit)).sum()


def find_root(X, y, cat=(), n_bins=4, method='width'):
    gains = {}
    for c in X.columns:
        codes = pd.factorize(X[c])[0] if c in cat else bin_feature(X[c], n_bins, method)[0]
        gains[c] = info_gain(codes, y)
    gains = dict(sorted(gains.items(), key=lambda kv: -kv[1]))
    return list(gains)[0], gains


class MyTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, n_bins=4, method='width', crit='entropy', cat=()):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.method = method
        self.crit = crit
        self.cat = cat

    def encode(self, X):
        cols = []
        for c in self.cols:
            if c in self.cat:
                cols.append(X[c].map(self.maps[c]).fillna(-1).astype(int).values)
            else:
                cols.append(np.digitize(X[c].values.astype(float), self.edges[c]))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.cols, self.maps, self.edges = list(X.columns), {}, {}
        for c in self.cols:
            if c in self.cat:
                self.maps[c] = {v: i for i, v in enumerate(X[c].unique())}
            else:
                self.edges[c] = bin_feature(X[c], self.n_bins, self.method)[1]
        self.classes_ = np.unique(y)
        self.tree = self.grow(self.encode(X), np.searchsorted(self.classes_, y), list(range(len(self.cols))), 0)
        return self

    def grow(self, C, y, free, depth):
        guess = np.bincount(y, minlength=len(self.classes_)).argmax()
        if len(set(y)) == 1 or not free or depth == self.max_depth:
            return guess
        gains = [info_gain(C[:, f], y, self.crit) for f in free]
        if max(gains) <= 0:
            return guess
        best = free[int(np.argmax(gains))]
        rest = [f for f in free if f != best]
        kids = {v: self.grow(C[C[:, best] == v], y[C[:, best] == v], rest, depth + 1) for v in np.unique(C[:, best])}
        return (best, guess, kids)

    def predict(self, X):
        out = []
        for row in self.encode(pd.DataFrame(X)):
            node = self.tree
            while isinstance(node, tuple):
                node = node[2].get(row[node[0]], node[1])
            out.append(node)
        return self.classes_[out]

    def show(self, node=None, indent='', levels=1):
        node = self.tree if node is None else node
        if not isinstance(node, tuple):
            print(indent + 'leaf ->', self.classes_[node])
        elif levels >= 0:
            print(indent + 'split on', self.cols[node[0]])
            for v, kid in node[2].items():
                print(indent + '  branch', v)
                self.show(kid, indent + '    ', levels - 1)


df = pd.read_csv('uniswap_v3_swaps_synthetic.csv')
df = df[df.fee_tier.isin([100, 10000])].reset_index(drop=True)
y = (df.fee_tier == 10000).astype(int).values
df['same_sender_recipient'] = (df.sender == df.recipient).astype(int)
cols = ['timestamp', 'amount0', 'amount1', 'amountUSD', 'token0_symbol', 'token1_symbol', 'token0_decimals',
        'token1_decimals', 'block_number', 'gas_used', 'gas_price', 'same_sender_recipient']
cat = ['token0_symbol', 'token1_symbol', 'token0_decimals', 'token1_decimals', 'same_sender_recipient']

print('A1 entropy:', entropy(y))
print('A2 gini:', gini(y))

Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.3, random_state=42, stratify=y)
Xtr, Xte = Xtr.copy(), Xte.copy()
for c in ['token0_symbol', 'token1_symbol']:
    mode = Xtr[c].mode()[0]
    counts = Xtr[c].value_counts()
    keep = counts[counts >= 300].index
    for d in (Xtr, Xte):
        d[c] = d[c].fillna(mode).where(lambda s: s.isin(keep), 'OTHER')
Ntr, Nte = Xtr.copy(), Xte.copy()
for c in ['token0_symbol', 'token1_symbol']:
    codes = {v: i for i, v in enumerate(sorted(Xtr[c].unique()))}
    Ntr[c], Nte[c] = Xtr[c].map(codes), Xte[c].map(codes)

print('\nA3/A4 root feature by information gain')
for method in ['width', 'frequency']:
    root, gains = find_root(Xtr, ytr, cat, 4, method)
    print(method, '->', root, {k: round(v, 5) for k, v in list(gains.items())[:4]})

print('\nA5 own tree')
for depth in [3, None]:
    t = MyTree(max_depth=depth, method='frequency', cat=cat).fit(Xtr, ytr)
    print('max_depth', depth, '| train', round((t.predict(Xtr) == ytr).mean(), 4), '| test', round((t.predict(Xte) == yte).mean(), 4))
    if depth == 3:
        t.show()

print('\nA6 tree picture -> tree.png')
sk3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42).fit(Ntr, ytr)
plt.figure(figsize=(16, 7))
plot_tree(sk3, feature_names=list(Ntr.columns), class_names=['fee 100', 'fee 10000'], filled=True, fontsize=8)
plt.savefig('tree.png', dpi=120)
plt.close()

print('\nA7 decision boundary -> boundary.png')
cont = [c for c in cols if c not in cat and Xtr[c].nunique() > 10]
f1, f2 = list(find_root(Xtr[cont], ytr, (), 4, 'frequency')[1])[:2]
slog = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
P = pd.DataFrame({f1: slog(Xtr[f1]), f2: slog(Xtr[f2])})
Pt = pd.DataFrame({f1: slog(Xte[f1]), f2: slog(Xte[f2])})
models = {'own tree (6 frequency bins)': MyTree(n_bins=6, method='frequency').fit(P, ytr),
          'sklearn tree (max_depth=4)': DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42).fit(P, ytr)}
gx, gy = np.meshgrid(np.linspace(P[f1].min() - .2, P[f1].max() + .2, 300), np.linspace(P[f2].min() - .2, P[f2].max() + .2, 300))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (name, m) in zip(axes, models.items()):
    z = m.predict(pd.DataFrame({f1: gx.ravel(), f2: gy.ravel()})).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=[-.5, .5, 1.5], colors=['#cfe0f7', '#fbd9cc'])
    ax.scatter(P[f1][:1500], P[f2][:1500], c=ytr[:1500], s=6, cmap=ListedColormap(['#2a78d6', '#eb6834']))
    ax.set_title('%s, test acc %.3f' % (name, (m.predict(Pt) == yte).mean()))
    ax.set_xlabel(f1 + ' (signed log)')
axes[0].set_ylabel(f2 + ' (signed log)')
plt.tight_layout()
plt.savefig('boundary.png', dpi=130)
plt.close()
print('features used:', f1, f2)

print('\nA8 GridSearchCV (5-fold)')
grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 8, 10, None],
        'min_samples_split': [2, 10, 30], 'min_samples_leaf': [1, 5, 20]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=5, n_jobs=-1).fit(Ntr, ytr)
print('sklearn tree best:', gs.best_params_, '| cv', round(gs.best_score_, 4), '| test', round(gs.score(Nte, yte), 4))
grid2 = {'max_depth': [2, 3, 4, 5, 6], 'n_bins': [3, 4, 6, 8], 'method': ['width', 'frequency'], 'crit': ['entropy', 'gini']}
gs2 = GridSearchCV(MyTree(cat=cat), grid2, cv=5).fit(Xtr, ytr)
print('own tree best:', gs2.best_params_, '| cv', round(gs2.best_score_, 4), '| test', round(gs2.score(Xte, yte), 4))
print('test baseline (majority class):', round(max(yte.mean(), 1 - yte.mean()), 4))
