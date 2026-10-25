import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')

df = pd.read_csv('uniswap_v3_swaps_synthetic.csv')
df = df[df.fee_tier.isin([100, 10000])].reset_index(drop=True)
y = (df.fee_tier == 10000).astype(int).values
df['same_sender_recipient'] = (df.sender == df.recipient).astype(int)
for col in ['amount0', 'amount1', 'amountUSD', 'gas_price']:
    df[col] = np.sign(df[col]) * np.log10(1 + np.abs(df[col]))

X = df[['timestamp', 'amount0', 'amount1', 'amountUSD', 'token0_symbol', 'token1_symbol', 'token0_decimals',
        'token1_decimals', 'block_number', 'gas_used', 'gas_price', 'same_sender_recipient']].copy()
for col in ['token0_symbol', 'token1_symbol']:
    X[col] = X[col].fillna(X[col].mode()[0])
    common = X[col].value_counts()
    X[col] = X[col].where(X[col].isin(common[common >= 300].index), 'OTHER')
X = pd.get_dummies(X, columns=['token0_symbol', 'token1_symbol'], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def evaluate(model):
    pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
    return {'train acc': accuracy_score(y_train, model.predict(X_train)),
            'test acc': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'f1': f1_score(y_test, pred),
            'roc auc': roc_auc_score(y_test, score)}


def make_pipe(model):
    return Pipeline([('scaler', StandardScaler()), ('clf', model)])


print('A1: perceptron and MLP with default settings')
perceptron = make_pipe(Perceptron(random_state=42))
mlp = make_pipe(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=42))
results = {}
for name, model in [('Perceptron', perceptron), ('MLP', mlp)]:
    model.fit(X_train, y_train)
    results[name] = evaluate(model)
a1 = pd.DataFrame(results).T.round(4)
print(a1)
a1.to_csv('a1_baselines.csv')

print('\nA2: RandomizedSearchCV for perceptron and MLP')
perceptron_params = {
    'clf__penalty': [None, 'l2', 'l1', 'elasticnet'],
    'clf__alpha': loguniform(1e-6, 1e-1),
    'clf__eta0': loguniform(1e-3, 10),
    'clf__max_iter': randint(200, 2000),
    'clf__class_weight': [None, 'balanced']}
mlp_params = {
    'clf__hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (64, 32)],
    'clf__activation': ['relu', 'tanh', 'logistic'],
    'clf__solver': ['adam', 'sgd'],
    'clf__alpha': loguniform(1e-6, 1e-1),
    'clf__learning_rate_init': loguniform(1e-4, 1e-1),
    'clf__batch_size': [32, 64, 128, 256]}

search_perceptron = RandomizedSearchCV(make_pipe(Perceptron(random_state=42)), perceptron_params, n_iter=25,
                                       scoring='f1', cv=5, random_state=42, n_jobs=-1)
search_mlp = RandomizedSearchCV(make_pipe(MLPClassifier(max_iter=500, early_stopping=True, random_state=42)),
                                mlp_params, n_iter=25, scoring='f1', cv=5, random_state=42, n_jobs=-1)
results = {}
for name, search in [('Perceptron (tuned)', search_perceptron), ('MLP (tuned)', search_mlp)]:
    search.fit(X_train, y_train)
    print(name, 'best cv f1: %.4f' % search.best_score_)
    print('  ', search.best_params_)
    results[name] = evaluate(search.best_estimator_)
a2 = pd.DataFrame(results).T.round(4)
print(a2)
a2.to_csv('a2_tuned.csv')

print('\nA3: comparing classifiers')
models = {
    'Perceptron (tuned)': search_perceptron.best_estimator_,
    'MLP (tuned)': search_mlp.best_estimator_,
    'SVM': make_pipe(SVC()),
    'Decision Tree': make_pipe(DecisionTreeClassifier(max_depth=8, random_state=42)),
    'Random Forest': make_pipe(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    'AdaBoost': make_pipe(AdaBoostClassifier(n_estimators=200, random_state=42)),
    'XGBoost': make_pipe(XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)),
    'CatBoost': make_pipe(CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)),
    'Naive Bayes': make_pipe(GaussianNB())}
results = {}
for name, model in models.items():
    if 'tuned' not in name:
        model.fit(X_train, y_train)
    results[name] = evaluate(model)
a3 = pd.DataFrame(results).T.sort_values('f1', ascending=False).round(4)
print(a3)
a3.to_csv('a3_classifier_comparison.csv')

a3[['test acc', 'precision', 'recall', 'f1', 'roc auc']].plot(kind='bar', figsize=(11, 5))
plt.title('Classifier comparison on the test set')
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('a3_comparison.png', dpi=140)
plt.close()

print('\nA4: pipeline with scaling and classifier, searched together')
pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))])
pipe_params = [
    {'clf': [RandomForestClassifier(random_state=42, n_jobs=-1)],
     'clf__n_estimators': randint(100, 500), 'clf__max_depth': [None, 6, 10, 16]},
    {'clf': [XGBClassifier(random_state=42)],
     'clf__n_estimators': randint(100, 500), 'clf__max_depth': randint(3, 9),
     'clf__learning_rate': loguniform(0.02, 0.3)},
    {'clf': [CatBoostClassifier(verbose=0, random_seed=42)],
     'clf__iterations': randint(150, 500), 'clf__depth': randint(4, 9)}]
pipe_search = RandomizedSearchCV(pipe, pipe_params, n_iter=15, scoring='f1', cv=5, random_state=42)
pipe_search.fit(X_train, y_train)
final_pipe = pipe_search.best_estimator_
print('best model:', type(final_pipe.named_steps['clf']).__name__)
print('best cv f1: %.4f' % pipe_search.best_score_)
a4 = pd.DataFrame({'final pipeline': evaluate(final_pipe)}).T.round(4)
print(a4)
a4.to_csv('a4_pipeline.csv')

print('\nA5: LIME explanations')
explainer = LimeTabularExplainer(X_train.values, feature_names=list(X_train.columns),
                                 class_names=['fee 100', 'fee 10000'], mode='classification', random_state=42)
pred = final_pipe.predict(X_test)
proba = final_pipe.predict_proba(X_test)[:, 1]

right_high = np.where((pred == 1) & (y_test == 1))[0]
right_low = np.where((pred == 0) & (y_test == 0))[0]
wrong = np.where(pred != y_test)[0]
cases = {'correct fee 10000': right_high[np.argmax(proba[right_high])],
         'correct fee 100': right_low[np.argmin(proba[right_low])],
         'misclassified': wrong[np.argmin(np.abs(proba[wrong] - 0.5))]}

rows = []
for number, (title, i) in enumerate(cases.items(), start=1):
    exp = explainer.explain_instance(X_test.iloc[i].values, final_pipe.predict_proba, num_features=8, top_labels=1)
    label = exp.available_labels()[0]
    print('\n%s: actual %d, predicted %d, P(fee 10000) = %.3f' % (title, y_test[i], pred[i], proba[i]))
    for feature, weight in exp.as_list(label=label):
        print('   %+.4f  %s' % (weight, feature))
        rows.append({'case': title, 'feature': feature, 'weight': weight})
    exp.as_pyplot_figure(label=label)
    plt.tight_layout()
    plt.savefig('a5_lime_%d.png' % number, dpi=140)
    plt.close()
    exp.save_to_file('a5_lime_%d.html' % number)
pd.DataFrame(rows).to_csv('a5_lime_explanations.csv', index=False)
