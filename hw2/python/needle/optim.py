"""Optimization module"""
import math

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        #u是列表是因为参数列表，用dict存储每个参数对应的u
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p is None:
                continue
            if p.grad is None:
                continue
            if p not in self.u:
                self.u[p] = ndl.Tensor.make_const(np.zeros(p.shape, dtype='float32'))
            grad_data = ndl.Tensor(p.grad.data,dtype="float32").data + self.weight_decay * p.data
            self.u[p] = self.momentum * self.u[p].data + (1 - self.momentum) * grad_data
            p.data = p.data - self.u[p] * self.lr
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p is None:
                continue
            elif p.grad is None:
                continue
            elif p not in self.m and p not in self.v:
                self.m[p] = ndl.Tensor.make_const(np.zeros(p.shape,dtype="float32"))
                self.v[p] = ndl.Tensor.make_const(np.zeros(p.shape,dtype="float32"))
            grad = ndl.Tensor(p.grad.data,dtype="float32").data+self.weight_decay*p.data
            self.m[p].data = self.beta1 * self.m[p].data + (1-self.beta1)*grad.data
            self.v[p].data = self.beta2 * self.v[p].data + (1-self.beta2)*grad.data**2
            assert((1-self.beta1**self.t)>0)
            assert ((1-self.beta2**self.t)>0)
            u_hat = self.m[p].data/(1-self.beta1**self.t)
            v_hat = self.v[p].data/(1-self.beta2**self.t)
            p.data = p.data - self.lr * u_hat/(v_hat**0.5+self.eps)

        ### END YOUR SOLUTION
