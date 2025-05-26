#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

# 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

# 활성화 함수와 손실 함수
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

# 배치 정규화 forward / backward
def batch_norm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, beta, eps)
    return out, cache

def batch_norm_backward(dout, cache):
    x, x_norm, mean, var, gamma, beta, eps = cache
    N, D = x.shape
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmean = np.sum(dx_norm * -1/np.sqrt(var + eps), axis=0) + dvar * np.mean(-2*(x-mean), axis=0)
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2*(x-mean)/N + dmean/N
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta

# 네트워크 초기화
input_size = 784
hidden_size1 = 256
hidden_size2 = 128
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

W3 = weight_init_std * np.random.randn(hidden_size2, output_size)
b3 = np.zeros(output_size)

# 하이퍼파라미터
iters_num = 10000
batch_size = 100
learning_rate = 0.01

train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    # 미니배치 선택
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Forward
    a1 = np.dot(x_batch, W1) + b1
    z1_bn, cache1 = batch_norm_forward(a1, gamma1, beta1)
    z1 = relu(z1_bn)

    a2 = np.dot(z1, W2) + b2
    z2_bn, cache2 = batch_norm_forward(a2, gamma2, beta2)
    z2 = relu(z2_bn)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    # Loss
    loss = cross_entropy(y, t_batch)
    train_loss_list.append(loss)

    # Backward
    dy = y.copy()
    dy[np.arange(batch_size), t_batch] -= 1
    dy /= batch_size

    dW3 = np.dot(z2.T, dy)
    db3 = np.sum(dy, axis=0)
    dz2 = np.dot(dy, W3.T)

    dz2_bn = dz2 * relu_derivative(z2_bn)
    dz2_bn, dgamma2, dbeta2 = batch_norm_backward(dz2_bn, cache2)

    dW2 = np.dot(z1.T, dz2_bn)
    db2 = np.sum(dz2_bn, axis=0)
    dz1 = np.dot(dz2_bn, W2.T)

    dz1_bn = dz1 * relu_derivative(z1_bn)
    dz1_bn, dgamma1, dbeta1 = batch_norm_backward(dz1_bn, cache1)

    dW1 = np.dot(x_batch.T, dz1_bn)
    db1 = np.sum(dz1_bn, axis=0)

    # 파라미터 갱신
    for param, dparam in zip(
        [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3],
        [dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2, dW3, db3]
    ):
        param -= learning_rate * dparam

    # 1 epoch마다 정확도 측정
    if i % iter_per_epoch == 0:
        # 학습 정확도
        a1 = np.dot(x_train, W1) + b1
        z1_bn, _ = batch_norm_forward(a1, gamma1, beta1)
        z1 = relu(z1_bn)

        a2 = np.dot(z1, W2) + b2
        z2_bn, _ = batch_norm_forward(a2, gamma2, beta2)
        z2 = relu(z2_bn)

        a3 = np.dot(z2, W3) + b3
        y_train = softmax(a3)
        train_acc = np.mean(np.argmax(y_train, axis=1) == t_train)

        # 테스트 정확도
        a1 = np.dot(x_test, W1) + b1
        z1_bn, _ = batch_norm_forward(a1, gamma1, beta1)
        z1 = relu(z1_bn)

        a2 = np.dot(z1, W2) + b2
        z2_bn, _ = batch_norm_forward(a2, gamma2, beta2)
        z2 = relu(z2_bn)

        a3 = np.dot(z2, W3) + b3
        y_test = softmax(a3)
        test_acc = np.mean(np.argmax(y_test, axis=1) == t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"epoch: {i//iter_per_epoch}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

# 결과 그래프
plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
