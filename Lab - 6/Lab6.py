import os
import json
import time
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs('outputs', exist_ok=True)

# ======================================================================
# PART 1 - MY OWN kNN (Lab 05 functions, student's own design)
# ======================================================================
# GenAI tool used: Claude - changed the dtype test so string columns are found on newer pandas
def encode_categorical(data):
    data = data.copy()
    for c in data.columns:
        if not pd.api.types.is_numeric_dtype(data[c]):
            data[c] = LabelEncoder().fit_transform(data[c].astype(str))
    return data

# GenAI tool used: Claude - text columns now always use the mode so this can run before encoding
def impute_missing_values(data, strategy='mean'):
    data = data.copy()
    for c in data.columns:
        if data[c].isnull().sum()>0:
            if not pd.api.types.is_numeric_dtype(data[c]):
                fv = data[c].mode()[0]
            else:
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



# ======================================================================
# PART 2 - GenAI kNN
# GenAI tool used: Claude (Anthropic) - whole of this part: distances, merge/quick/heap sort, kNN class with fit/predict/score and weighted voting
# ======================================================================
def gen_fill_missing(train, test, how="median"):
    train = train.copy()
    test = test.copy()
    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            if how == "mean":
                val = train[col].mean()
            elif how == "mode":
                val = train[col].mode()[0]
            else:
                val = train[col].median()
        else:
            val = train[col].mode()[0]
        train[col] = train[col].fillna(val)
        test[col] = test[col].fillna(val)
    return train, test


def gen_encode_data(train, test, cat_cols):
    train = train.copy()
    test = test.copy()
    for col in cat_cols:
        codes = {c: i for i, c in enumerate(sorted(train[col].unique()))}
        train[col] = train[col].map(codes)
        test[col] = test[col].map(codes).fillna(-1)
    return train, test


def gen_scale_data(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


def gen_get_distance(a, B, mode="euclidean"):
    if mode == "euclidean":
        return np.sqrt(((B - a) ** 2).sum(axis=1))
    if mode == "manhattan":
        return np.abs(B - a).sum(axis=1)
    if mode == "chebyshev":
        return np.abs(B - a).max(axis=1)
    if mode == "cosine":
        top = B @ a
        bottom = np.linalg.norm(B, axis=1) * np.linalg.norm(a)
        bottom[bottom == 0] = 1e-12
        return 1 - top / bottom
    raise ValueError("unknown distance mode: " + mode)


def gen_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = gen_merge_sort(arr[:mid])
    right = gen_merge_sort(arr[mid:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def gen_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    smaller = [x for x in arr if x < pivot]
    same = [x for x in arr if x == pivot]
    bigger = [x for x in arr if x > pivot]
    return gen_quick_sort(smaller) + same + gen_quick_sort(bigger)


def gen_heap_sort(arr):
    arr = list(arr)
    n = len(arr)

    def sift(root, end):
        while 2 * root + 1 <= end:
            child = 2 * root + 1
            if child + 1 <= end and arr[child] < arr[child + 1]:
                child += 1
            if arr[root] < arr[child]:
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                break

    for start in range(n // 2 - 1, -1, -1):
        sift(start, n - 1)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sift(0, end - 1)
    return arr


GEN_SORTS = {"merge": gen_merge_sort, "quick": gen_quick_sort, "heap": gen_heap_sort, "builtin": sorted}


class GenAIKNN:
    def __init__(self, k=3, metric="euclidean", sort_type="merge", weighted=False):
        self.k = k
        self.metric = metric
        self.sort_type = sort_type
        self.weighted = weighted

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def nearest(self, X, kmax):
        dist = np.zeros((len(X), kmax))
        idx = np.zeros((len(X), kmax), dtype=int)
        sorter = GEN_SORTS[self.sort_type]
        for n, row in enumerate(X):
            d = gen_get_distance(row, self.X, self.metric)
            pairs = sorter(list(zip(d.tolist(), range(len(d)))))[:kmax]
            dist[n] = [p[0] for p in pairs]
            idx[n] = [p[1] for p in pairs]
        return dist, idx

    def vote(self, dists, labels):
        scores = {}
        for d, lab in zip(dists, labels):
            w = 1 / (d + 1e-9) if self.weighted else 1
            scores[lab] = scores.get(lab, 0) + w
        top = max(scores.values())
        tied = [c for c in scores if abs(scores[c] - top) < 1e-12 * max(1, top)]
        if len(tied) == 1:
            return tied[0]
        for lab in labels:
            if lab in tied:
                return lab

    def predict_from(self, dist, idx, k):
        return np.array([self.vote(dist[i][:k], self.y[idx[i][:k]]) for i in range(len(dist))])

    def predict(self, X, k=None):
        k = self.k if k is None else k
        dist, idx = self.nearest(np.asarray(X, dtype=float), k)
        return self.predict_from(dist, idx, k)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


# ======================================================================
# PART 3 - Dataset preparation
# GenAI tool used: Claude (Anthropic) - builds the two-class dataset (fee tier 100 vs 10000) from the Uniswap data
# ======================================================================
def prepare_dataset(path='uniswap_v3_swaps_synthetic.csv'):
    df = pd.read_csv(path)
    df = df[df['fee_tier'].isin([100, 10000])].reset_index(drop=True)
    df['label'] = (df['fee_tier'] == 10000).astype(int)
    df['same_sender_recipient'] = (df['sender'] == df['recipient']).astype(int)
    cols = ['timestamp', 'amount0', 'amount1', 'amountUSD', 'token0_symbol', 'token1_symbol',
            'token0_decimals', 'token1_decimals', 'block_number', 'gas_used', 'gas_price',
            'same_sender_recipient', 'label']
    df = df[cols]
    df.to_csv('project_dataset.csv', index=False)
    return df


# ======================================================================
# PART 4 - Unit and functional tests (Lab 05 and Lab 06 functions)
# GenAI tool used: Claude (Anthropic) - all test cases
# ======================================================================
class TestEncodeCategorical(unittest.TestCase):
    def test_text_column_becomes_numeric(self):
        out = encode_categorical(pd.DataFrame({'a': ['x', 'y', 'x'], 'b': [1, 2, 3]}))
        self.assertTrue(pd.api.types.is_numeric_dtype(out['a']))

    def test_same_text_same_code(self):
        out = encode_categorical(pd.DataFrame({'a': ['x', 'y', 'x']}))
        self.assertEqual(out['a'][0], out['a'][2])
        self.assertNotEqual(out['a'][0], out['a'][1])

    def test_numeric_column_untouched(self):
        df = pd.DataFrame({'b': [1.5, 2.5, 3.5]})
        pd.testing.assert_frame_equal(encode_categorical(df), df)

    def test_input_not_modified(self):
        df = pd.DataFrame({'a': ['x', 'y']})
        encode_categorical(df)
        self.assertEqual(df['a'][0], 'x')

    def test_genai_encoder_unseen_category_is_minus_one(self):
        tr = pd.DataFrame({'c': ['a', 'b']})
        te = pd.DataFrame({'c': ['a', 'z']})
        _, te_enc = gen_encode_data(tr, te, ['c'])
        self.assertEqual(list(te_enc['c']), [0, -1])


class TestImputation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 5.0], 'c': ['u', 'u', 'v', None]})

    def test_mean(self):
        out = impute_missing_values(self.df[['a']], 'mean')
        self.assertAlmostEqual(out['a'][2], (1 + 2 + 5) / 3)

    def test_median(self):
        self.assertEqual(impute_missing_values(self.df[['a']], 'median')['a'][2], 2.0)

    def test_mode(self):
        df = pd.DataFrame({'a': [1.0, 1.0, 4.0, np.nan]})
        self.assertEqual(impute_missing_values(df, 'mode')['a'][3], 1.0)

    def test_text_column_uses_mode(self):
        out = impute_missing_values(self.df, 'mean')
        self.assertEqual(out['c'][3], 'u')

    def test_no_missing_left(self):
        self.assertEqual(impute_missing_values(self.df, 'median').isna().sum().sum(), 0)

    def test_genai_fill_missing_uses_train_statistics_for_test(self):
        tr = pd.DataFrame({'a': [1.0, 3.0]})
        te = pd.DataFrame({'a': [np.nan]})
        _, te_f = gen_fill_missing(tr, te, how='mean')
        self.assertEqual(te_f['a'][0], 2.0)


class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        self.assertAlmostEqual(calculate_distance([0, 0], [3, 4], 'euclidean'), 5.0)

    def test_manhattan(self):
        self.assertAlmostEqual(calculate_distance([0, 0], [3, 4], 'manhattan'), 7.0)

    def test_minkowski_p3(self):
        self.assertAlmostEqual(calculate_distance([0, 0], [1, 1], 'minkowski'), 2 ** (1 / 3))

    def test_zero_distance_to_self(self):
        self.assertEqual(calculate_distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_symmetry(self):
        a, b = [1, 5, 2], [4, 1, 9]
        self.assertEqual(calculate_distance(a, b), calculate_distance(b, a))

    def test_genai_matches_own_euclidean(self):
        a = np.array([1.0, 2.0, 3.0])
        B = np.array([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
        expected = [calculate_distance(a, r) for r in B]
        np.testing.assert_allclose(gen_get_distance(a, B, 'euclidean'), expected)

    def test_genai_manhattan_chebyshev_cosine(self):
        a, B = np.array([1.0, 0.0]), np.array([[0.0, 1.0]])
        self.assertAlmostEqual(gen_get_distance(a, B, 'manhattan')[0], 2.0)
        self.assertAlmostEqual(gen_get_distance(a, B, 'chebyshev')[0], 1.0)
        self.assertAlmostEqual(gen_get_distance(a, B, 'cosine')[0], 1.0)

    def test_genai_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            gen_get_distance(np.zeros(2), np.zeros((1, 2)), 'nope')


class TestSorting(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.d = list(rng.randint(0, 20, 60).astype(float))
        self.idx = list(range(60))

    def test_three_sorts_order_distances(self):
        for algo in ['bubble', 'insertion', 'merge']:
            sd, _ = sort_distances(self.d, self.idx, algo)
            self.assertEqual(list(sd), sorted(self.d), algo)

    def test_indices_follow_distances(self):
        for algo in ['bubble', 'insertion', 'merge']:
            sd, si = sort_distances(self.d, self.idx, algo)
            for dist, i in zip(sd, si):
                self.assertEqual(dist, self.d[i], algo)

    def test_empty_and_single(self):
        for algo in ['bubble', 'insertion', 'merge']:
            self.assertEqual(list(sort_distances([], [], algo)[0]), [])
            self.assertEqual(list(sort_distances([2.0], [0], algo)[0]), [2.0])

    def test_already_sorted_and_reversed(self):
        for algo in ['bubble', 'insertion', 'merge']:
            self.assertEqual(list(sort_distances([1, 2, 3], [0, 1, 2], algo)[0]), [1, 2, 3])
            self.assertEqual(list(sort_distances([3, 2, 1], [0, 1, 2], algo)[0]), [1, 2, 3])

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(KeyError):
            sort_distances([1], [0], 'bogo')

    def test_input_lists_not_modified(self):
        d, i = [3.0, 1.0, 2.0], [0, 1, 2]
        sort_distances(d, i, 'bubble')
        self.assertEqual(d, [3.0, 1.0, 2.0])

    def test_genai_sorts_match_builtin(self):
        pairs = [(float(v), i) for i, v in enumerate(self.d)]
        for name in ['merge', 'quick', 'heap']:
            self.assertEqual(GEN_SORTS[name](pairs), sorted(pairs), name)


class TestNeighboursAndVoting(unittest.TestCase):
    def test_k_nearest_without_ties(self):
        sd, si = [1.0, 2.0, 3.0, 4.0], [5, 6, 7, 8]
        y = {5: 0, 6: 1, 7: 1, 8: 0}
        ni, nd = get_k_neighbors(sd, si, 2, y)
        self.assertEqual(ni, [5, 6])
        self.assertEqual(nd, [1.0, 2.0])

    def test_boundary_tie_returns_exactly_k(self):
        sd, si = [1.0, 2.0, 2.0, 2.0, 9.0], [0, 1, 2, 3, 4]
        y = np.array([0, 1, 1, 0, 0])
        ni, nd = get_k_neighbors(sd, si, 2, y)
        self.assertEqual(len(ni), 2)
        self.assertEqual(len(nd), 2)

    def test_boundary_tie_prefers_more_common_class(self):
        sd, si = [1.0, 2.0, 2.0, 2.0], [0, 1, 2, 3]
        y = np.array([0, 1, 1, 0])
        ni, _ = get_k_neighbors(sd, si, 2, y)
        self.assertEqual(ni[0], 0)
        self.assertEqual(ni[1], 1)

    def test_majority_vote(self):
        self.assertEqual(majority_vote([1, 1, 0]), 1)

    def test_majority_vote_tie_picks_smaller_label(self):
        self.assertEqual(majority_vote([1, 0]), 0)

    def test_weighted_vote_closer_neighbour_wins(self):
        self.assertEqual(weighted_vote([0, 1, 1], [0.1, 5.0, 5.0]), 0)

    def test_weighted_vote_exact_match_dominates(self):
        self.assertEqual(weighted_vote([1, 0, 0], [0.0, 1.0, 1.0]), 1)

    def test_weighted_equals_majority_when_distances_equal(self):
        self.assertEqual(weighted_vote([1, 1, 0], [2.0, 2.0, 2.0]), majority_vote([1, 1, 0]))


class TestMyKNN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_stores_data(self):
        m = knn_fit(self.X, self.y)
        np.testing.assert_array_equal(m['X_train'], self.X)
        np.testing.assert_array_equal(m['y_train'], self.y)

    def test_predict_separable_points(self):
        m = knn_fit(self.X, self.y)
        np.testing.assert_array_equal(knn_predict(m, [[0.5, 0.5], [10.5, 10.5]], k=3), [0, 1])

    def test_predict_returns_one_label_per_row(self):
        m = knn_fit(self.X, self.y)
        self.assertEqual(len(knn_predict(m, self.X, k=1)), len(self.X))

    def test_k1_on_training_data_is_perfect(self):
        m = knn_fit(self.X, self.y)
        self.assertEqual(knn_score(m, self.X, self.y, k=1), 1.0)

    def test_score_is_between_0_and_1(self):
        m = knn_fit(self.X, self.y)
        s = knn_score(m, [[5, 5]], [0], k=3)
        self.assertTrue(0.0 <= s <= 1.0)

    def test_score_known_value(self):
        m = knn_fit(self.X, self.y)
        self.assertEqual(knn_score(m, [[0.5, 0.5], [10.5, 10.5]], [0, 0], k=3), 0.5)

    def test_weighted_runs_and_agrees_on_clear_cases(self):
        m = knn_fit(self.X, self.y)
        np.testing.assert_array_equal(knn_predict(m, [[0.2, 0.2]], k=3, weighted=True), [0])

    def test_all_sort_algorithms_give_same_predictions(self):
        m = knn_fit(self.X, self.y)
        q = [[2, 2], [8, 8], [5, 5]]
        base = knn_predict(m, q, k=3, sort_algorithm='merge')
        for algo in ['bubble', 'insertion']:
            np.testing.assert_array_equal(knn_predict(m, q, k=3, sort_algorithm=algo), base)

    def test_all_metrics_run(self):
        m = knn_fit(self.X, self.y)
        for metric in ['euclidean', 'manhattan', 'minkowski']:
            self.assertEqual(len(knn_predict(m, [[1, 1]], k=3, metric=metric)), 1)

    def test_scale_features_uses_train_statistics(self):
        tr, te = scale_features(np.array([[0.0], [2.0]]), np.array([[1.0]]))
        self.assertAlmostEqual(tr.mean(), 0.0)
        self.assertAlmostEqual(te[0, 0], 0.0)


class TestGenAIKNN(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(120, 4))
        self.y = (self.X[:, 0] + 0.3 * rng.normal(size=120) > 0).astype(int)
        self.Q = rng.normal(size=(25, 4))

    def test_matches_sklearn_unweighted(self):
        mine = GenAIKNN(k=5).fit(self.X, self.y).predict(self.Q)
        ref = KNeighborsClassifier(5).fit(self.X, self.y).predict(self.Q)
        np.testing.assert_array_equal(mine, ref)

    def test_matches_sklearn_weighted(self):
        mine = GenAIKNN(k=5, weighted=True).fit(self.X, self.y).predict(self.Q)
        ref = KNeighborsClassifier(5, weights='distance').fit(self.X, self.y).predict(self.Q)
        np.testing.assert_array_equal(mine, ref)

    def test_sort_choice_does_not_change_predictions(self):
        base = GenAIKNN(k=3, sort_type='builtin').fit(self.X, self.y).predict(self.Q)
        for s in ['merge', 'quick', 'heap']:
            np.testing.assert_array_equal(GenAIKNN(k=3, sort_type=s).fit(self.X, self.y).predict(self.Q), base)

    def test_score_matches_manual_accuracy(self):
        m = GenAIKNN(k=3).fit(self.X, self.y)
        self.assertAlmostEqual(m.score(self.Q, np.zeros(25, dtype=int)), float(np.mean(m.predict(self.Q) == 0)))

    def test_vote_tie_goes_to_nearest_neighbour(self):
        m = GenAIKNN(k=2).fit(np.array([[1.0], [-1.5]]), np.array([0, 1]))
        self.assertEqual(m.predict(np.array([[0.0]]))[0], 0)

    def test_equal_distance_uses_lower_index(self):
        m = GenAIKNN(k=1).fit(np.array([[1.0], [-1.0]]), np.array([1, 0]))
        self.assertEqual(m.predict(np.array([[0.0]]))[0], 1)


class TestFunctionalPipeline(unittest.TestCase):
    def test_end_to_end_two_class_dataset(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({'f1': np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)],
                           'f2': np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)],
                           'cat': ['a'] * 60 + ['b'] * 60,
                           'label': [0] * 60 + [1] * 60})
        df.loc[3, 'f1'] = np.nan
        df = encode_categorical(impute_missing_values(df, 'median'))
        self.assertEqual(df.isna().sum().sum(), 0)
        X, y = df.drop('label', axis=1).values, df['label'].values
        idx = rng.permutation(len(X))
        tr, te = idx[:84], idx[84:]
        Xtr, Xte = scale_features(X[tr], X[te])
        model = knn_fit(Xtr, y[tr])
        self.assertGreaterEqual(knn_score(model, Xte, y[te], k=3), 0.95)

    def test_custom_versions_and_sklearn_agree_on_dataset_sample(self):
        rng = np.random.RandomState(5)
        X = rng.normal(size=(150, 5))
        y = (X[:, 1] > 0).astype(int)
        Xtr, ytr, Xte = X[:100], y[:100], X[100:]
        a = knn_predict(knn_fit(Xtr, ytr), Xte, k=3)
        b = GenAIKNN(k=3).fit(Xtr, ytr).predict(Xte)
        c = KNeighborsClassifier(3).fit(Xtr, ytr).predict(Xte)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, c)




def run_unit_tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__('__main__'))
    with open('outputs/unit_test_log.txt', 'w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print('unit tests run: %d | failures: %d | errors: %d' % (result.testsRun, len(result.failures), len(result.errors)))
    return result


# ======================================================================
# PART 5 - Experiments: A1 (Lab 05 repeated) and A3 (three-way comparison)
# GenAI tool used: Claude (Anthropic) - experiment code, built on the functions above
# ======================================================================
def main():
    RESULTS = {}

    data = impute_missing_values(prepare_dataset())
    data = encode_categorical(data)
    X, y = data.drop('label', axis=1).values, data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = scale_features(X_train, X_test)
    RESULTS['train_shape'], RESULTS['test_shape'] = list(X_train.shape), list(X_test.shape)
    print('A3 (split): X_train', X_train.shape, 'X_test', X_test.shape)


    def take_sample(n_train, n_test, seed=42):
        rng = np.random.RandomState(seed)
        tr = rng.choice(len(X_train), size=n_train, replace=False)
        te = rng.choice(len(X_test), size=n_test, replace=False)
        return X_train[tr], y_train[tr], X_test[te], y_test[te]


    print('\n===== A1: Lab 05 experiments repeated =====')
    Xtr_s, ytr_s, Xte_s, yte_s = take_sample(300, 100)

    neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    print('A5: sklearn accuracy (full test set)', round(neigh.score(X_test, y_test), 4))
    print('A6: predictions', neigh.predict(X_test)[:20])

    custom_model = knn_fit(Xtr_s, ytr_s)
    a7 = knn_score(custom_model, Xte_s, yte_s, k=3)
    a9 = knn_score(custom_model, Xte_s, yte_s, k=3, weighted=True)
    print('A7: custom accuracy (300 train / 100 test sample)', a7)
    print('A9: weighted accuracy (same sample)', a9)
    RESULTS['A1'] = {'sklearn_k3_full': float(neigh.score(X_test, y_test)), 'custom_k3_sample': float(a7),
                     'weighted_k3_sample': float(a9)}

    print('\nsorting algorithms give the same predictions? (150 train / 25 test)')
    Xa, ya, Xb, yb = take_sample(150, 25, seed=1)
    m_small = knn_fit(Xa, ya)
    sort_preds = {s: knn_predict(m_small, Xb, 3, sort_algorithm=s) for s in ['bubble', 'insertion', 'merge']}
    same = all((sort_preds[s] == sort_preds['merge']).all() for s in sort_preds)
    print(' bubble / insertion / merge identical:', same)
    RESULTS['A1']['three_sorts_identical'] = bool(same)

    k_values = list(range(1, 12, 2))
    rows = []
    for k in k_values:
        sk_full = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
        sk_sub = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ytr_s).score(Xte_s, yte_s)
        cu = knn_score(custom_model, Xte_s, yte_s, k=k)
        wt = knn_score(custom_model, Xte_s, yte_s, k=k, weighted=True)
        rows.append({'k': k, 'sklearn_full_data': sk_full, 'sklearn_same_sample': sk_sub, 'custom': cu, 'custom_weighted': wt})
        print('k=', k, 'sklearn(full)=', round(sk_full, 4), 'sklearn(sample)=', round(sk_sub, 4),
              'custom=', round(cu, 4), 'weighted=', round(wt, 4))
    sweep = pd.DataFrame(rows)
    sweep.to_csv('outputs/a1_accuracy_vs_k.csv', index=False)
    RESULTS['A1']['sweep'] = rows

    BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e6e5e1', 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    plt.figure(figsize=(8, 5))
    plt.plot(sweep.k, sweep.sklearn_full_data, marker='o', color=BLUE, lw=2, label='sklearn kNN (full train/test)')
    plt.plot(sweep.k, sweep.sklearn_same_sample, marker='D', color=YELLOW, lw=2, ls=':', label='sklearn kNN (same 300/100 sample)')
    plt.plot(sweep.k, sweep.custom, marker='s', color=ORANGE, lw=2, label='Custom kNN')
    plt.plot(sweep.k, sweep.custom_weighted, marker='^', color=AQUA, lw=2, label='Custom weighted kNN')
    plt.xlabel('k')
    plt.xticks(k_values)
    plt.ylabel('Accuracy')
    plt.title('Lab 6 / A1: accuracy vs k')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('outputs/accuracy_comparison.png', dpi=150)
    plt.close()


    print('\n===== A2: unit and functional tests =====')
    run_unit_tests()

    print('\n===== A3: three versions of kNN =====')
    N_TRAIN, N_TEST, K, RUNS = 2000, 500, 3, 10
    Xtr, ytr, Xte, yte = take_sample(N_TRAIN, N_TEST, seed=7)
    print('sample: %d train / %d test, k=%d, %d timed runs each' % (N_TRAIN, N_TEST, K, RUNS))


    def run_user():
        model = knn_fit(Xtr, ytr)
        return knn_predict(model, Xte, k=K)


    def run_sklearn():
        model = KNeighborsClassifier(n_neighbors=K).fit(Xtr, ytr)
        return model.predict(Xte)


    def run_genai():
        model = GenAIKNN(k=K, metric='euclidean', sort_type='merge').fit(Xtr, ytr)
        return model.predict(Xte)


    versions = [('My own kNN (Part 1)', run_user),
                ('Scikit-Learn KNeighborsClassifier', run_sklearn),
                ('GenAI kNN (Part 2, Claude)', run_genai)]

    table, preds = [], {}
    for name, fn in versions:
        times = []
        for r in range(RUNS):
            t0 = time.perf_counter()
            p = fn()
            times.append(time.perf_counter() - t0)
        preds[name] = p
        table.append({'version': name,
                      'accuracy': accuracy_score(yte, p),
                      'precision': precision_score(yte, p, zero_division=0),
                      'recall': recall_score(yte, p, zero_division=0),
                      'f1': f1_score(yte, p, zero_division=0),
                      'avg_time_s': float(np.mean(times)),
                      'std_time_s': float(np.std(times)),
                      'min_time_s': float(np.min(times)),
                      'max_time_s': float(np.max(times))})
        print(name, {k: round(v, 4) for k, v in table[-1].items() if k != 'version'}, flush=True)

    res = pd.DataFrame(table)
    res.to_csv('outputs/a3_comparison.csv', index=False)
    RESULTS['A3'] = {'config': {'n_train': N_TRAIN, 'n_test': N_TEST, 'k': K, 'runs': RUNS}, 'rows': table}

    names = [n for n, _ in versions]
    agree = {}
    for i in range(3):
        for j in range(i + 1, 3):
            agree['%s  vs  %s' % (names[i], names[j])] = float((preds[names[i]] == preds[names[j]]).mean())
    RESULTS['A3']['prediction_agreement'] = agree
    print('\nprediction agreement between versions:')
    for k_, v_ in agree.items():
        print(' ', k_, round(v_, 4))


    json.dump(RESULTS, open('outputs/lab06_results.json', 'w'), indent=2, default=str)
    print('\nDONE')


if __name__ == '__main__':
    main()
