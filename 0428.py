#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

# 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

def get_batch(x, t, batch_size):
    idx = np.random.choice(x.shape[0], batch_size)
    return x[idx], t[idx]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy(y, t):
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 하이퍼파라미터
input_size = 784
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 64
output_size = 10

weight_init_std = 0.01
W1 = weight_init_std * np.random.randn(input_size, hidden_size1)
b1 = np.zeros(hidden_size1)
gamma1 = np.ones(hidden_size1)
beta1 = np.zeros(hidden_size1)

W2 = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
b2 = np.zeros(hidden_size2)
gamma2 = np.ones(hidden_size2)
beta2 = np.zeros(hidden_size2)

W3 = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
b3 = np.zeros(hidden_size3)
gamma3 = np.ones(hidden_size3)
beta3 = np.zeros(hidden_size3)

W4 = weight_init_std * np.random.randn(hidden_size3, output_size)
b4 = np.zeros(output_size)

# 학습 설정
iters_num = 10000
batch_size = 100
learning_rate = 0.01

train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

epsilon = 1e-7

for i in range(iters_num):
    x_batch, t_batch = get_batch(x_train, t_train, batch_size)

    # forward
    # 1층
    z1 = x_batch @ W1 + b1
    mu1 = np.mean(z1, axis=0)
    var1 = np.var(z1, axis=0)
    z1_norm = (z1 - mu1) / np.sqrt(var1 + epsilon)
    z1_bn = gamma1 * z1_norm + beta1
    a1 = relu(z1_bn)

    # 2층
    z2 = a1 @ W2 + b2
    mu2 = np.mean(z2, axis=0)
    var2 = np.var(z2, axis=0)
    z2_norm = (z2 - mu2) / np.sqrt(var2 + epsilon)
    z2_bn = gamma2 * z2_norm + beta2
    a2 = relu(z2_bn)

    # 3층
    z3 = a2 @ W3 + b3
    mu3 = np.mean(z3, axis=0)
    var3 = np.var(z3, axis=0)
    z3_norm = (z3 - mu3) / np.sqrt(var3 + epsilon)
    z3_bn = gamma3 * z3_norm + beta3
    a3 = relu(z3_bn)

    # 출력층
    scores = a3 @ W4 + b4
    y = softmax(scores)

    loss = cross_entropy(y, t_batch)
    train_loss_list.append(loss)

    # backward
    batch_size_inv = 1.0 / batch_size

    dy = (y - np.eye(output_size)[t_batch]) * batch_size_inv  # one-hot
    dW4 = a3.T @ dy
    db4 = np.sum(dy, axis=0)

    da3 = dy @ W4.T
    dz3_bn = da3 * relu_derivative(z3_bn)
    dgamma3 = np.sum(dz3_bn * z3_norm, axis=0)
    dbeta3 = np.sum(dz3_bn, axis=0)
    dz3_norm = dz3_bn * gamma3
    dvar3 = np.sum(dz3_norm * (z3 - mu3) * -0.5 * (var3 + epsilon)**(-1.5), axis=0)
    dmu3 = np.sum(dz3_norm * -1/np.sqrt(var3 + epsilon), axis=0) + dvar3 * np.mean(-2*(z3 - mu3), axis=0)
    dz3 = dz3_norm / np.sqrt(var3 + epsilon) + dvar3 * 2*(z3 - mu3)/batch_size + dmu3/batch_size
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ W3.T
    dz2_bn = da2 * relu_derivative(z2_bn)
    dgamma2 = np.sum(dz2_bn * z2_norm, axis=0)
    dbeta2 = np.sum(dz2_bn, axis=0)
    dz2_norm = dz2_bn * gamma2
    dvar2 = np.sum(dz2_norm * (z2 - mu2) * -0.5 * (var2 + epsilon)**(-1.5), axis=0)
    dmu2 = np.sum(dz2_norm * -1/np.sqrt(var2 + epsilon), axis=0) + dvar2 * np.mean(-2*(z2 - mu2), axis=0)
    dz2 = dz2_norm / np.sqrt(var2 + epsilon) + dvar2 * 2*(z2 - mu2)/batch_size + dmu2/batch_size
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1_bn = da1 * relu_derivative(z1_bn)
    dgamma1 = np.sum(dz1_bn * z1_norm, axis=0)
    dbeta1 = np.sum(dz1_bn, axis=0)
    dz1_norm = dz1_bn * gamma1
    dvar1 = np.sum(dz1_norm * (z1 - mu1) * -0.5 * (var1 + epsilon)**(-1.5), axis=0)
    dmu1 = np.sum(dz1_norm * -1/np.sqrt(var1 + epsilon), axis=0) + dvar1 * np.mean(-2*(z1 - mu1), axis=0)
    dz1 = dz1_norm / np.sqrt(var1 + epsilon) + dvar1 * 2*(z1 - mu1)/batch_size + dmu1/batch_size
    dW1 = x_batch.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # 파라미터 갱신
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    gamma1 -= learning_rate * dgamma1
    beta1 -= learning_rate * dbeta1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    gamma2 -= learning_rate * dgamma2
    beta2 -= learning_rate * dbeta2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    gamma3 -= learning_rate * dgamma3
    beta3 -= learning_rate * dbeta3

    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    # 1 epoch마다 정확도 측정
    if i % iter_per_epoch == 0:
        # 학습 데이터
        a1 = relu(gamma1 * ((x_train @ W1 + b1 - np.mean(x_train @ W1 + b1, axis=0)) / np.sqrt(np.var(x_train @ W1 + b1, axis=0) + epsilon)) + beta1)
        a2 = relu(gamma2 * ((a1 @ W2 + b2 - np.mean(a1 @ W2 + b2, axis=0)) / np.sqrt(np.var(a1 @ W2 + b2, axis=0) + epsilon)) + beta2)
        a3 = relu(gamma3 * ((a2 @ W3 + b3 - np.mean(a2 @ W3 + b3, axis=0)) / np.sqrt(np.var(a2 @ W3 + b3, axis=0) + epsilon)) + beta3)
        scores = a3 @ W4 + b4
        y_train = softmax(scores)
        train_acc = np.mean(np.argmax(y_train, axis=1) == t_train)

        # 테스트 데이터
        a1 = relu(gamma1 * ((x_test @ W1 + b1 - np.mean(x_test @ W1 + b1, axis=0)) / np.sqrt(np.var(x_test @ W1 + b1, axis=0) + epsilon)) + beta1)
        a2 = relu(gamma2 * ((a1 @ W2 + b2 - np.mean(a1 @ W2 + b2, axis=0)) / np.sqrt(np.var(a1 @ W2 + b2, axis=0) + epsilon)) + beta2)
        a3 = relu(gamma3 * ((a2 @ W3 + b3 - np.mean(a2 @ W3 + b3, axis=0)) / np.sqrt(np.var(a2 @ W3 + b3, axis=0) + epsilon)) + beta3)
        scores = a3 @ W4 + b4
        y_test = softmax(scores)
        test_acc = np.mean(np.argmax(y_test, axis=1) == t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"epoch {int(i/iter_per_epoch)} - train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

# 결과
plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
