"""The module.
"""
import math
import random
from typing import List, Callable, Any
import numpy as np
import xlwings

from needle.autograd import Tensor
import needle.init as init
import needle.ops as ops


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    #获得所有子式的参数集合
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    #获得所有的子模型
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,nonlinearity="relu"))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features,1,nonlinearity="relu").reshape((1,-1)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.matmul(self.weight)+self.bias.broadcast_to((X.shape[0],self.out_features))
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        temp=ops.relu(x)
        return temp
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x=m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_max = logits.shape[1]
        y_one_hot = init.one_hot(y_max,y)
        #去最大值化的logsum函数
        log_sum_z = ops.logsumexp(logits,axes=(1,)).sum()
        #logits 乘与 y_one_hot向量得到z_y的值
        z_y = (ops.EWiseMul()(logits,y_one_hot)).sum()
        return (log_sum_z-z_y)/logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim,requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim,requires_grad=True))
        self.running_mean = Parameter(init.zeros(self.dim,requires_grad=True))
        self.running_var = Parameter(init.ones(self.dim,requires_grad=True))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            x_mean = x.sum(axes=(0,))/x.shape[0]
            #注意这里是对一列(batch)求平均值，所以要对x_mean reshape到一行排(1,x.shape[1]),
            #再broad_cast到x.shape，这样就是x-每一列的均值
            x_minus_miu =x-(x_mean.reshape((1,x.shape[1]))).broadcast_to(x.shape)
            x_var = (x_minus_miu**2).sum(axes=(0,))/x.shape[0]
            x_hat = self.weight.broadcast_to(x.shape)*(x_minus_miu/(x_var.reshape((1,x.shape[1])).broadcast_to(x.shape)+self.eps)**0.5)+self.bias.broadcast_to(x.shape)

            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*x_mean
            self.running_var = (1-self.momentum)*self.running_var+self.momentum*x_var
            return x_hat
        else:
            self.running_mean = self.running_mean.reshape((1,x.shape[1]))
            self.running_var = self.running_var.reshape((1,x.shape[1]))
            x_norm = (x-self.running_mean.broadcast_to(x.shape))/((self.running_var.broadcast_to(x.shape)+self.eps)**0.5)
            return self.weight.broadcast_to(x.shape)*x_norm+self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim,requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #Remember to broadcast_to!
        x_mean = x.sum(axes=(1,)).reshape((x.shape[0],1))/x.shape[1]
        x_minus_mean = x-x_mean.broadcast_to(x.shape)
        x_std = (((x_minus_mean)**2).sum(axes=(1,)).reshape((x.shape[0],1))/x.shape[1]+self.eps)**0.5
        return self.weight.broadcast_to(x.shape)*(x_minus_mean/x_std.broadcast_to(x.shape))+self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #构造随机变量prob(x.shape[0]*x.shape[1]),p<=1-p则随机变量为1
        #所以x = x*prob/(1-p)
        if self.training:
            prob = Parameter(init.randb(x.shape[0],x.shape[1], p=1-self.p))
            x = x*prob/(1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return  self.fn(x)+x
        ### END YOUR SOLUTION



