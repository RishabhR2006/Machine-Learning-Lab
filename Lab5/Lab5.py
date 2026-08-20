import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def encode_categorical(data):
    data = data.copy()
    for c in data.columns:
        if data[c].dtype==object:
            data[c] = LabelEncoder().fit_transform(data[c].astype(str))
    return data

def impute_missing_values(data, strategy='mean'):
    data = data.copy()
    for c in data.columns:
        if data[c].isnull().sum()>0:
            fv = data[c].mean() if strategy=='mean' else data[c].median() if strategy=='median' else data[c].mode()[0]
            data[c] = data[c].fillna(fv)
    return data

def calculate_distance(a, b, metric='euclidean'):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if metric=='euclidean': return np.sqrt(np.sum((a-b)**2))
    if metric=='manhattan': return np.sum(np.abs(a-b))
    p = 3
    return np.sum(np.abs(a-b)**p)**(1/p)

def bubble_sort(d, idx):
    d, idx = d.copy(), idx.copy()
    n = len(d)
    for i in range(n-1):
        for j in range(n-1-i):
            if d[j]>d[j+1]:
                d[j], d[j+1], idx[j], idx[j+1] = d[j+1], d[j], idx[j+1], idx[j]
    return d, idx

def insertion_sort(d, idx):
    d, idx = d.copy(), idx.copy()
    for i in range(1, len(d)):
        kd, ki, j = d[i], idx[i], i-1
        while j>=0 and d[j]>kd:
            d[j+1], idx[j+1] = d[j], idx[j]
            j -= 1
        d[j+1], idx[j+1] = kd, ki
    return d, idx

def merge_sort(d, idx):
    d, idx = list(d), list(idx)
    if len(d)<=1: return d, idx
    mid = len(d)//2
    ld, li = merge_sort(d[:mid], idx[:mid])
    rd, ri = merge_sort(d[mid:], idx[mid:])
    md, mi, i, j = [], [], 0, 0
    while i<len(ld) and j<len(rd):
        if ld[i]<=rd[j]: md.append(ld[i]); mi.append(li[i]); i += 1
        else: md.append(rd[j]); mi.append(ri[j]); j += 1
    return md+ld[i:]+rd[j:], mi+li[i:]+ri[j:]

def sort_distances(d, idx, algorithm='merge'):
    return {'bubble': bubble_sort, 'insertion': insertion_sort, 'merge': merge_sort}[algorithm](d, idx)

def get_k_neighbors(sd, si, k, y_train):
    boundary = sd[k-1]
    ei, ed = list(si[:k]), list(sd[:k])
    i = k
    while i<len(sd) and sd[i]==boundary:
        ei.append(si[i]); ed.append(sd[i]); i += 1
    if len(ei)>k:
        start = ed.index(boundary)
        tied = ei[start:]
        counts = {}
        for idx in tied: counts[y_train[idx]] = counts.get(y_train[idx], 0)+1
        tied_sorted = sorted(tied, key=lambda idx: (-counts[y_train[idx]], idx))
        needed = k-start
        ei = ei[:start]+tied_sorted[:needed]
        ed = ed[:start]+[boundary]*needed
    return ei[:k], ed[:k]

def majority_vote(labels):
    counts = {}
    for l in labels: counts[l] = counts.get(l, 0)+1
    m = max(counts.values())
    return min(l for l, c in counts.items() if c==m)

def weighted_vote(labels, dists):
    weights = {}
    for l, d in zip(labels, dists): weights[l] = weights.get(l, 0)+1/(d+1e-8)
    m = max(weights.values())
    return min(l for l, w in weights.items() if w==m)

def knn_fit(X_train, y_train):
    return {'X_train': np.array(X_train), 'y_train': np.array(y_train)}

def knn_predict(model, X_test, k=3, metric='euclidean', sort_algorithm='merge', weighted=False):
    X_train, y_train = model['X_train'], model['y_train']
    preds = []
    for tp in np.array(X_test):
        d = [calculate_distance(tp, xp, metric) for xp in X_train]
        idx = list(range(len(d)))
        sd, si = sort_distances(d, idx, sort_algorithm)
        ni, nd = get_k_neighbors(sd, si, k, y_train)
        nl = [y_train[i] for i in ni]
        preds.append(weighted_vote(nl, nd) if weighted else majority_vote(nl))
    return np.array(preds)

def knn_score(model, X_test, y_test, k=3, metric='euclidean', sort_algorithm='merge', weighted=False):
    preds = knn_predict(model, X_test, k, metric, sort_algorithm, weighted)
    return np.sum(preds==np.array(y_test))/len(y_test)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

if __name__=='__main__':
    data = impute_missing_values(encode_categorical(pd.read_csv('project_dataset.csv')))
    X, y = data.drop('label', axis=1).values, data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = scale_features(X_train, X_test)

    rng = np.random.RandomState(42)
    tr_idx = rng.choice(len(X_train), size=300, replace=False)
    te_idx = rng.choice(len(X_test), size=100, replace=False)
    X_train_s, y_train_s = X_train[tr_idx], y_train[tr_idx]
    X_test_s, y_test_s = X_test[te_idx], y_test[te_idx]

    print('A3: X_train', X_train.shape, 'X_test', X_test.shape)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    print('A5: sklearn accuracy', neigh.score(X_test, y_test))
    print('A6: predictions', neigh.predict(X_test)[:20])

    custom_model = knn_fit(X_train_s, y_train_s)
    print('A7: custom accuracy', knn_score(custom_model, X_test_s, y_test_s, k=3))
    print('A9: weighted accuracy', knn_score(custom_model, X_test_s, y_test_s, k=3, weighted=True))

    k_values = list(range(1, 12, 2))
    sk_acc, cu_acc, wt_acc = [], [], []
    for k in k_values:
        sk = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        sk_acc.append(sk.score(X_test, y_test))
        cu_acc.append(knn_score(custom_model, X_test_s, y_test_s, k=k))
        wt_acc.append(knn_score(custom_model, X_test_s, y_test_s, k=k, weighted=True))
        print('k=', k, 'sklearn=', sk_acc[-1], 'custom=', cu_acc[-1], 'weighted=', wt_acc[-1])

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sk_acc, marker='o', label='sklearn kNN')
    plt.plot(k_values, cu_acc, marker='s', label='Custom kNN')
    plt.plot(k_values, wt_acc, marker='^', label='Custom Weighted kNN')
    plt.xlabel('k'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig('accuracy_comparison.png')