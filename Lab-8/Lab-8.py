import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']


def summation(w, x):
    return w[..., 0] + np.dot(w[..., 1:], x)


def step(z):
    return np.where(z >= 0, 1.0, 0.0)


def bipolar_step(z):
    return np.where(z >= 0, 1.0, -1.0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z, slope=0.01):
    return np.where(z > 0, z, slope * z)


def comparator(target, output):
    return target - output


ACTIVATIONS = {'step': step, 'bipolar step': bipolar_step, 'sigmoid': sigmoid, 'tanh': tanh,
               'relu': relu, 'leaky relu': leaky_relu}

X_GATE = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
AND = np.array([0, 0, 0, 1], dtype=float)
XOR = np.array([0, 1, 1, 0], dtype=float)
START_WEIGHTS = [10, 0.2, -0.75]


def targets_for(act_name, t):
    t = np.asarray(t, dtype=float)
    return 2 * t - 1 if act_name == 'bipolar step' else t


def train(X, T, W, lr, act, max_epochs=1000, goal=0.002):
    W = np.array(W, dtype=float)
    T = np.asarray(T, dtype=float).reshape(len(X), -1)
    W = W.reshape(T.shape[1], -1)
    errors = []
    for epoch in range(1, max_epochs + 1):
        for x, t in zip(X, T):
            out = act(summation(W, x))
            W = W + lr * np.outer(comparator(t, out), np.append(1, x))
        sse = sum(np.sum(comparator(t, act(summation(W, x))) ** 2) for x, t in zip(X, T))
        errors.append(sse)
        if sse <= goal:
            return W, errors, epoch
    return W, errors, None


def describe(epochs, errors):
    if epochs:
        return 'converged in %d epochs (final error %.5f)' % (epochs, errors[-1])
    return 'did not converge in %d epochs (final error %.4f)' % (len(errors), errors[-1])


def plot_errors(curves, title, filename):
    plt.figure(figsize=(8, 5))
    for i, (name, errors) in enumerate(curves.items()):
        plt.plot(range(1, len(errors) + 1), errors, label=name, color=COLORS[i % 5], lw=2)
    plt.yscale('symlog', linthresh=0.01)
    plt.xlabel('epoch')
    plt.ylabel('sum-square error (log scale)')
    plt.title(title, loc='left', fontweight='bold')
    plt.grid(True, color='#e6e5e1')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def sigmoid_net(X, t, lr=0.05, max_epochs=1000, goal=0.002):
    hidden = np.array([[0.1, 0.2, -0.3], [-0.2, 0.4, 0.1]])
    output = np.array([0.3, 0.5, -0.4])
    errors = []
    for epoch in range(1, max_epochs + 1):
        for x, target in zip(X, t):
            xb = np.append(1, x)
            h = sigmoid(hidden @ xb)
            hb = np.append(1, h)
            o = sigmoid(output @ hb)
            delta_o = (target - o) * o * (1 - o)
            delta_h = delta_o * output[1:] * h * (1 - h)
            output = output + lr * delta_o * hb
            hidden = hidden + lr * np.outer(delta_h, xb)
        preds = np.array([sigmoid(output @ np.append(1, sigmoid(hidden @ np.append(1, x)))) for x in X])
        sse = np.sum((t - preds) ** 2)
        errors.append(sse)
        if sse <= goal:
            return errors, epoch, preds
    return errors, None, preds


print('A1: units')
z = np.array([-2.0, 0.0, 2.0])
for name, f in ACTIVATIONS.items():
    print('  %-13s f(%s) = %s' % (name, z, np.round(f(z), 4)))
print('  summation([1, 2, 3], [4, 5]) =', summation(np.array([1.0, 2, 3]), np.array([4.0, 5])))
print('  comparator(1, 0.3) =', comparator(1, 0.3))

print('\nA2: AND gate, step activation, lr 0.05, start weights', START_WEIGHTS)
w, errors, epochs = train(X_GATE, AND, START_WEIGHTS, 0.05, step)
print('  ', describe(epochs, errors))
print('   learned weights:', np.round(w.ravel(), 3))
print('   outputs:', step(np.array([summation(w, x) for x in X_GATE])).ravel())
plot_errors({'step': errors}, 'AND gate with step activation (lr 0.05)', 'a2_and_step.png')

print('\nA3: AND gate with other activations, lr 0.05')
curves = {'step': errors}
for name in ['bipolar step', 'sigmoid', 'relu']:
    w, e, ep = train(X_GATE, targets_for(name, AND), START_WEIGHTS, 0.05, ACTIVATIONS[name])
    curves[name] = e
    print('  %-13s %s' % (name, describe(ep, e)))
plot_errors(curves, 'AND gate: error per epoch for each activation (lr 0.05)', 'a3_and_activations.png')

print('\nA4: AND gate, step activation, different learning rates')
rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
needed = []
for lr in rates:
    w, e, ep = train(X_GATE, AND, START_WEIGHTS, lr, step)
    needed.append(ep if ep else 1000)
    print('  lr %.1f -> %s' % (lr, describe(ep, e)))
plt.figure(figsize=(8, 5))
plt.plot(rates, needed, marker='o', color=COLORS[0], lw=2)
plt.xlabel('learning rate')
plt.ylabel('epochs to converge')
plt.title('AND gate: epochs needed vs learning rate', loc='left', fontweight='bold')
plt.grid(True, color='#e6e5e1')
plt.tight_layout()
plt.savefig('a4_learning_rates.png', dpi=140)
plt.close()

print('\nA5: XOR gate, lr 0.05 (A2 and A3 repeated)')
curves = {}
for name in ['step', 'bipolar step', 'sigmoid', 'relu']:
    w, e, ep = train(X_GATE, targets_for(name, XOR), START_WEIGHTS, 0.05, ACTIVATIONS[name])
    curves[name] = e
    print('  %-13s %s' % (name, describe(ep, e)))
plot_errors(curves, 'XOR gate: a single perceptron cannot get the error to zero', 'a5_xor.png')

print('\nA6: customer data, sigmoid perceptron')
customers = np.array([[20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393], [19, 1, 2, 110], [24, 4, 2, 280],
                      [22, 1, 5, 167], [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148], [16, 2, 4, 198]], dtype=float)
high = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)
items = customers[:, :3]
scaled = (items - items.mean(axis=0)) / items.std(axis=0)
w6, e6, ep6 = train(scaled, high, [0.1, 0.1, 0.1, 0.1], 0.5, sigmoid, max_epochs=20000)
probs = sigmoid(np.array([summation(w6, x) for x in scaled])).ravel()
print('  ', describe(ep6, e6))
print('   weights (bias, candies, mangoes, milk):', np.round(w6.ravel(), 3))
print('   accuracy: %.0f%%' % (100 * np.mean((probs >= 0.5) == high)))

print('\nA7: perceptron vs matrix pseudo-inverse')
design = np.c_[np.ones(len(scaled)), scaled]
w7 = np.linalg.pinv(design) @ high
scores = design @ w7
print('   pseudo-inverse weights:', np.round(w7, 3))
print('   pseudo-inverse accuracy: %.0f%%' % (100 * np.mean((scores >= 0.5) == high)))
table = pd.DataFrame({'customer': ['C%d' % i for i in range(1, 11)], 'actual': high.astype(int),
                      'perceptron': np.round(probs, 3), 'pinv score': np.round(scores, 3)})
print(table.to_string(index=False))
raw_design = np.c_[np.ones(len(items)), items]
print('   pseudo-inverse on raw items predicting Payment (bias, candy, mango, milk):',
      np.round(np.linalg.pinv(raw_design) @ customers[:, 3], 3))

print('\nA8: 2-2-1 sigmoid network with back-propagation, AND gate, lr 0.05')
e8, ep8, p8 = sigmoid_net(X_GATE, AND)
print('  ', describe(ep8, e8))
e8_long, ep8_long, p8_long = sigmoid_net(X_GATE, AND, max_epochs=200000)
print('   with the limit lifted to 200000 epochs:', describe(ep8_long, e8_long))
print('   outputs:', np.round(p8_long, 3))

print('\nA9: same network, XOR gate')
e9, ep9, p9 = sigmoid_net(X_GATE, XOR)
print('  ', describe(ep9, e9))
e9_long, ep9_long, p9_long = sigmoid_net(X_GATE, XOR, max_epochs=200000)
print('   with the limit lifted to 200000 epochs:', describe(ep9_long, e9_long))
print('   outputs:', np.round(p9_long, 3))
plot_errors({'AND': e8, 'XOR': e9}, 'Back-propagation network: error per epoch (lr 0.05)', 'a8_a9_backprop.png')

print('\nA10: AND gate with two output nodes, lr 0.05')
two_out = np.array([[1, 0] if t == 0 else [0, 1] for t in AND], dtype=float)
curves = {}
for name in ['step', 'bipolar step', 'sigmoid', 'relu']:
    T = 2 * two_out - 1 if name == 'bipolar step' else two_out
    w, e, ep = train(X_GATE, T, [START_WEIGHTS, START_WEIGHTS], 0.05, ACTIVATIONS[name])
    curves[name] = e
    picked = [int(np.argmax(ACTIVATIONS[name](summation(w, x)))) for x in X_GATE]
    print('  %-13s %s | predicted gate output: %s' % (name, describe(ep, e), picked))
plot_errors(curves, 'AND gate with two output nodes: error per epoch', 'a10_two_outputs.png')

print('\nA11: scikit-learn MLPClassifier on the gates')
for gate_name, gate in [('AND', AND), ('XOR', XOR)]:
    mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='lbfgs', max_iter=2000, random_state=1)
    mlp.fit(X_GATE, gate)
    print('  %s: predictions %s | accuracy %.0f%%' % (gate_name, mlp.predict(X_GATE), 100 * mlp.score(X_GATE, gate)))

print('\nA12: MLPClassifier on the project dataset (fee tier 100 vs 10000)')
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
scaler = StandardScaler().fit(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=42)
mlp.fit(scaler.transform(X_train), y_train)
pred = mlp.predict(scaler.transform(X_test))
print('   train accuracy %.4f | test accuracy %.4f | majority baseline %.4f' % (
    mlp.score(scaler.transform(X_train), y_train), accuracy_score(y_test, pred), max(y_test.mean(), 1 - y_test.mean())))
print('   precision %.4f | recall %.4f | f1 %.4f' % (
    precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)))
