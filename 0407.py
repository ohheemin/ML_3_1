# coding: utf-8

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.functions import *
from common.gradient import numerical_gradient

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from collections import OrderedDict

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

#========================================================================================

def cross_entropy_error(y, t):
    if y.ndim == 1: y = y.reshape(1, -1)
    if t.ndim == 1: t = t.reshape(1, -1)
    if t.size == y.size: t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

#==================================================================================

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)


    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()

    return grad

    # for idx in range(x.size):
    #     tmp_val = x.flat[idx]
    #     x.flat[idx] = tmp_val + h
    #     fxh1 = f(x)
    #     x.flat[idx] = tmp_val - h
    #     fxh2 = f(x)
    #     grad.flat[idx] = (fxh1 - fxh2) / (2 * h)
    #     x.flat[idx] = tmp_val
    # return grad

#=============================================================================================

def accuracy(y, t):
    y = np.argmax(y, axis=1)
    if t.ndim != 1: t = np.argmax(t, axis=1)
    return np.sum(y == t) / float(y.shape[0])

#=============================================================================================

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
#==========================================================================================

    def predict(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = np.maximum(0, a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
#================================================================================

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

#==============================================================================

    # def gradient(self, x, t):
    #     grads = {}
    #     batch_num = x.shape[0]

    #     W1, b1 = self.params['W1'], self.params['b1']
    #     W2, b2 = self.params['W2'], self.params['b2']
    #     a1 = np.dot(x, W1) + b1
    #     z1 = np.maximum(0, a1)
    #     a2 = np.dot(z1, W2) + b2
    #     y = softmax(a2)

    #     dy = (y - t) / batch_num
    #     grads['W2'] = np.dot(z1.T, dy)
    #     grads['b2'] = np.sum(dy, axis=0)

    #     da1 = np.dot(dy, W2.T)
    #     dz1 = da1 * (a1 > 0)
    #     grads['W1'] = np.dot(x.T, dz1)
    #     grads['b1'] = np.sum(dz1, axis=0)

    #     return grads

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def gradient(self, x, t):
        grads = {}
        batch_num = x.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = np.maximum(0, a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = da1 * (a1 > 0)
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.05

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

train_loss_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if (i + 1) % 1000 == 0:
        print(f"Iteration {i + 1}: Loss = {loss:.4f}")

y_test = network.predict(x_test)
test_acc = accuracy(y_test, t_test)
print(f"\n ************************************* \n ** << Test Accuracy: ( {test_acc * 100:.2f}% ) >> ** \n ************************************* ")


plt.plot(train_loss_list, color='red')
plt.title("<< Loss and Iteration graph >> ")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim(0, ) 
plt.grid(True)
plt.show()
#plt.ylim(0, 6) 

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
