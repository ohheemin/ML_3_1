import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

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

def accuracy(x, t, W1, b1, W2, b2, activation_func):
    a1 = activation_func(x @ W1 + b1)
    y = softmax(a1 @ W2 + b2)
    y_label = np.argmax(y, axis=1)
    t_label = np.argmax(t, axis=1)
    return np.mean(y_label == t_label)

# 2-layer 신경망 학습 함수
def train_nn_2layer(activation_func=relu): 
    np.random.seed(42)
    input_dim = 784
    hidden_dim = 100
    output_dim = 10
    learning_rate = 0.1
    epochs = 5
    batch_size = 100

    # 가중치 초기화
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))

    train_loss_list = []
    train_acc_list = []

    # 학습 루프
    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        loss_epoch = 0

        for i in range(0, x_train.shape[0], batch_size):
            batch_mask = perm[i:i + batch_size]
            x_batch = x_train[batch_mask]
            y_batch = t_train[batch_mask]

            # Forward
            z1 = x_batch @ W1 + b1
            a1 = activation_func(z1)
            z2 = a1 @ W2 + b2
            y = softmax(z2)
            loss = cross_entropy(y, y_batch)
            loss_epoch += loss

            # Backward
            batch_size_actual = x_batch.shape[0]
            dL_dz2 = (y - y_batch) / batch_size_actual
            dL_dW2 = a1.T @ dL_dz2
            dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

            dL_da1 = dL_dz2 @ W2.T
            if activation_func == relu:
                dL_dz1 = dL_da1 * relu_grad(z1)
            else:
                dL_dz1 = dL_da1 * sigmoid_grad(z1)
            dL_dW1 = x_batch.T @ dL_dz1
            dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

            # Update
            W1 -= learning_rate * dL_dW1
            b1 -= learning_rate * dL_db1
            W2 -= learning_rate * dL_dW2
            b2 -= learning_rate * dL_db2

        # 에폭마다 출력
        acc = accuracy(x_test, t_test, W1, b1, W2, b2, activation_func)
        train_loss_list.append(loss_epoch)
        train_acc_list.append(acc)

    return train_loss_list, train_acc_list

# 3-layer 신경망 학습 함수
def train_nn_3layer(activation_func=relu):  # 함수 객체를 매개변수로 전달
    np.random.seed(42)
    input_dim = 784
    hidden1_dim = 100
    hidden2_dim = 50
    output_dim = 10
    learning_rate = 0.1
    epochs = 30
    batch_size = 100

    # 가중치 초기화
    W1 = np.random.randn(input_dim, hidden1_dim) * 0.01
    b1 = np.zeros((1, hidden1_dim))
    W2 = np.random.randn(hidden1_dim, hidden2_dim) * 0.01
    b2 = np.zeros((1, hidden2_dim))
    W3 = np.random.randn(hidden2_dim, output_dim) * 0.01
    b3 = np.zeros((1, output_dim))

    train_loss_list = []
    train_acc_list = []

    # 학습 루프
    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        loss_epoch = 0

        for i in range(0, x_train.shape[0], batch_size):
            batch_mask = perm[i:i + batch_size]
            x_batch = x_train[batch_mask]
            y_batch = t_train[batch_mask]

            # Forward
            z1 = x_batch @ W1 + b1
            a1 = activation_func(z1)
            z2 = a1 @ W2 + b2
            a2 = activation_func(z2)
            z3 = a2 @ W3 + b3
            y = softmax(z3)
            loss = cross_entropy(y, y_batch)
            loss_epoch += loss

            # Backward
            batch_size_actual = x_batch.shape[0]
            dL_dz3 = (y - y_batch) / batch_size_actual
            dL_dW3 = a2.T @ dL_dz3
            dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

            dL_da2 = dL_dz3 @ W3.T
            dL_dz2 = dL_da2 * relu_grad(z2) if activation_func == relu else dL_da2 * sigmoid_grad(z2)
            dL_dW2 = a1.T @ dL_dz2
            dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

            dL_da1 = dL_dz2 @ W2.T
            dL_dz1 = dL_da1 * relu_grad(z1) if activation_func == relu else dL_da1 * sigmoid_grad(z1)
            dL_dW1 = x_batch.T @ dL_dz1
            dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

            # Update
            W1 -= learning_rate * dL_dW1
            b1 -= learning_rate * dL_db1
            W2 -= learning_rate * dL_dW2
            b2 -= learning_rate * dL_db2
            W3 -= learning_rate * dL_dW3
            b3 -= learning_rate * dL_db3

        # 에폭마다 출력
            acc = accuracy(x_test, t_test, W1, b1, W2, b2, W3, b3, activation_func)
        train_loss_list.append(loss_epoch)
        train_acc_list.append(acc)

    return train_loss_list, train_acc_list

# 학습 (2-layer & 3-layer NN with ReLU and Sigmoid)
relu_loss_2, relu_acc_2 = train_nn_2layer(relu)
sigmoid_loss_2, sigmoid_acc_2 = train_nn_2layer(sigmoid)
relu_loss_3, relu_acc_3 = train_nn_3layer(relu)
sigmoid_loss_3, sigmoid_acc_3 = train_nn_3layer(sigmoid)

# 그래프 출력
plt.figure(figsize=(12, 6))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(relu_loss_2, label='ReLU 2-layer')
plt.plot(sigmoid_loss_2, label='Sigmoid 2-layer')
plt.plot(relu_loss_3, label='ReLU 3-layer')
plt.plot(sigmoid_loss_3, label='Sigmoid 3-layer')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(relu_acc_2, label='ReLU 2-layer')
plt.plot(sigmoid_acc_2, label='Sigmoid 2-layer')
plt.plot(relu_acc_3, label='ReLU 3-layer')
plt.plot(sigmoid_acc_3, label='Sigmoid 3-layer')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


