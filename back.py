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

def accuracy(x, t, params, activation, layer):
    W1, b1, W2, b2 = params[:4]
    if layer == 2:
        z1 = x @ W1 + b1
        a1 = relu(z1) if activation == 'relu' else sigmoid(z1)
        y = softmax(a1 @ W2 + b2)
    else:
        W3, b3 = params[4:]
        z1 = x @ W1 + b1
        a1 = relu(z1) if activation == 'relu' else sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2) if activation == 'relu' else sigmoid(z2)
        y = softmax(a2 @ W3 + b3)
    y_label = np.argmax(y, axis=1)
    t_label = np.argmax(t, axis=1)
    return np.mean(y_label == t_label)

def train_network(layer=2, activation='relu', epochs=50, batch_size=100, learning_rate=0.1):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    input_dim = 784
    hidden1_dim = 100
    hidden2_dim = 50
    output_dim = 10

    np.random.seed(43)
    W1 = np.random.randn(input_dim, hidden1_dim) * 0.01
    b1 = np.zeros((1, hidden1_dim))
    W2 = np.random.randn(hidden1_dim, hidden2_dim if layer == 3 else output_dim) * 0.01
    b2 = np.zeros((1, hidden2_dim if layer == 3 else output_dim))

    if layer == 3:
        W3 = np.random.randn(hidden2_dim, output_dim) * 0.01
        b3 = np.zeros((1, output_dim))
        params = [W1, b1, W2, b2, W3, b3]
    else:
        params = [W1, b1, W2, b2]

    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        loss_epoch = 0

        for i in range(0, x_train.shape[0], batch_size):
            batch_mask = perm[i:i+batch_size]
            x_batch = x_train[batch_mask]
            y_batch = t_train[batch_mask]

            z1 = x_batch @ params[0] + params[1]
            a1 = relu(z1) if activation == 'relu' else sigmoid(z1)

            if layer == 2:
                z2 = a1 @ params[2] + params[3]
                y = softmax(z2)
            else:
                z2 = a1 @ params[2] + params[3]
                a2 = relu(z2) if activation == 'relu' else sigmoid(z2)
                z3 = a2 @ params[4] + params[5]
                y = softmax(z3)

            loss = cross_entropy(y, y_batch)
            loss_epoch += loss

            batch_size_actual = x_batch.shape[0]
            dL_dz = (y - y_batch) / batch_size_actual

            if layer == 2:
                dL_dW2 = a1.T @ dL_dz
                dL_db2 = np.sum(dL_dz, axis=0, keepdims=True)

                dL_da1 = dL_dz @ params[2].T
                dL_dz1 = dL_da1 * (relu_grad(z1) if activation == 'relu' else sigmoid_grad(z1))
                dL_dW1 = x_batch.T @ dL_dz1
                dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

                params[0] -= learning_rate * dL_dW1
                params[1] -= learning_rate * dL_db1
                params[2] -= learning_rate * dL_dW2
                params[3] -= learning_rate * dL_db2

            else:  
                dL_dW3 = a2.T @ dL_dz
                dL_db3 = np.sum(dL_dz, axis=0, keepdims=True)

                dL_da2 = dL_dz @ params[4].T
                dL_dz2 = dL_da2 * (relu_grad(z2) if activation == 'relu' else sigmoid_grad(z2))
                dL_dW2 = a1.T @ dL_dz2
                dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

                dL_da1 = dL_dz2 @ params[2].T
                dL_dz1 = dL_da1 * (relu_grad(z1) if activation == 'relu' else sigmoid_grad(z1))
                dL_dW1 = x_batch.T @ dL_dz1
                dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

                # Update
                params[0] -= learning_rate * dL_dW1
                params[1] -= learning_rate * dL_db1
                params[2] -= learning_rate * dL_dW2
                params[3] -= learning_rate * dL_db2
                params[4] -= learning_rate * dL_dW3
                params[5] -= learning_rate * dL_db3

        acc = accuracy(x_test, t_test, params, activation, layer)
        print(f"[{activation.upper()} {layer}-layer] Epoch {epoch+1:2d} | Loss: {loss_epoch:.4f} | Acc: {acc:.4f}")
        loss_list.append(loss_epoch)
        acc_list.append(acc)

    return loss_list, acc_list



results = {}
for layer in [2, 3]:
    for act in ['relu', 'sigmoid']:
        key = f"{act}_layer{layer}"
        results[key] = train_network(layer=layer, activation=act)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for k in results:
    plt.plot(results[k][0], label=f"{k}")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
for k in results:
    plt.plot(results[k][1], label=f"{k}")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
