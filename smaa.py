import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

def accuracy(x, t, params, activation):
    W1, b1, W2, b2, W3, b3 = params
    z1 = x @ W1 + b1
    a1 = relu(z1) if activation == 'relu' else sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2) if activation == 'relu' else sigmoid(z2)
    y = softmax(a2 @ W3 + b3)
    y_label = np.argmax(y, axis=1)
    t_label = np.argmax(t, axis=1)
    return np.mean(y_label == t_label)

def adam_update(param, dparam, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v 

def train_network(activation='relu', epochs=20, batch_size=100, learning_rate=0.01, optimizer='sgd'):  # 값이 변하지 않고 유려하게 되는데 그 이유?
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    input_dim = x_train.shape[1]      
    hidden1_dim = 100                   
    hidden2_dim = 50                   
    output_dim = t_train.shape[1]       

    np.random.seed(43)
    W1 = np.random.randn(input_dim, hidden1_dim) * 0.01
    b1 = np.zeros((1, hidden1_dim))
    W2 = np.random.randn(hidden1_dim, hidden2_dim) * 0.01
    b2 = np.zeros((1, hidden2_dim))
    W3 = np.random.randn(hidden2_dim, output_dim) * 0.01
    b3 = np.zeros((1, output_dim))

    params = [W1, b1, W2, b2, W3, b3]

    if optimizer == 'momentum':
        v = [np.zeros_like(p) for p in params]
        momentum = 0.9
    elif optimizer == 'adagrad':
        h = [np.zeros_like(p) for p in params]
        epsilon = 1e-7
    elif optimizer == 'adam':
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        epsilon = 1e-8

    loss_list, acc_list = [], []

    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        loss_epoch = 0

        for i in range(0, x_train.shape[0], batch_size):
            batch_mask = perm[i:i+batch_size]
            x_batch = x_train[batch_mask]
            y_batch = t_train[batch_mask]
            z1 = x_batch @ params[0] + params[1]
            a1 = relu(z1) if activation == 'relu' else sigmoid(z1)
            z2 = a1 @ params[2] + params[3]
            a2 = relu(z2) if activation == 'relu' else sigmoid(z2)
            z3 = a2 @ params[4] + params[5]
            y = softmax(z3)

            loss = cross_entropy(y, y_batch)
            loss_epoch += loss
            batch_size_actual = x_batch.shape[0]
            dL_dz = (y - y_batch) / batch_size_actual

            dW3 = a2.T @ dL_dz
            db3 = np.sum(dL_dz, axis=0, keepdims=True)
            da2 = dL_dz @ params[4].T
            dz2 = da2 * (relu_grad(z2) if activation == 'relu' else sigmoid_grad(z2))

            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ params[2].T
            dz1 = da1 * (relu_grad(z1) if activation == 'relu' else sigmoid_grad(z1))

            dW1 = x_batch.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            grads = [dW1, db1, dW2, db2, dW3, db3]

            if optimizer == 'sgd':
                for i in range(6):
                    params[i] -= learning_rate * grads[i]

            elif optimizer == 'momentum':
                for i in range(6):
                    v[i] = momentum * v[i] - learning_rate * grads[i]
                    params[i] += v[i]

            elif optimizer == 'adagrad':
                for i in range(6):
                    h[i] += grads[i] ** 2
                    params[i] -= learning_rate * grads[i] / (np.sqrt(h[i]) + epsilon)

            elif optimizer == 'adam':
                t = epoch * (x_train.shape[0] // batch_size) + (i // batch_size) + 1
                for i in range(6):
                    params[i], m[i], v[i] = adam_update(params[i], grads[i], m[i], v[i], t, learning_rate, epsilon=epsilon)

        acc = accuracy(x_test, t_test, params, activation)
        print(f"[{optimizer.upper()} {activation}] Epoch {epoch+1:2d} | Loss: {loss_epoch:.4f} | Acc: {acc:.4f}")
        loss_list.append(loss_epoch)
        acc_list.append(acc)

    return loss_list, acc_list

results = {}

for optimizer in ['sgd', 'momentum', 'adagrad', 'adam']:
    results[optimizer] = train_network(activation='relu', optimizer=optimizer)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

for k in results:
    plt.plot(results[k][0], label=k)

plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)

for k in results:
    plt.plot(results[k][1], label=k)

plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
