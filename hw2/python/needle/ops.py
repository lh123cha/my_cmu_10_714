"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * array_api.power(node.inputs[0],self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/node.inputs[1] , -1*out_grad*node.inputs[0]/(node.inputs[1])**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes==None:
            return array_api.swapaxes(a,-1,-2)
        else:
            return array_api.swapaxes(a,self.axes[0],self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes==None:
            return out_grad.transpose((-1,-2))
        else:
            return out_grad.transpose((self.axes[1],self.axes[0]))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X=node.inputs[0]
        return out_grad.reshape(X.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        broadcast_dim = []
        input_dim = len(x.shape)

        output_dim = len(out_grad.shape)
        # ???????????????????????????????????????:1*3 ??? 3*3
        # ????????????????????????????????????out_grad??????????????????1*3???????????????
        # 3*1 ---> 3*3??????????????????out_grad????????????????????????????????????3*1???????????????
        if input_dim == output_dim:
            for i in range(input_dim):
                if x.shape[i] != out_grad.shape[i]:
                    broadcast_dim.append(i)
        # ???????????????????????????????????????:3*3 ---> 3*3*6
        # ???????????????????????????????????????????????????????????????boardcast_dim????????????????????????????????????boardcast_dim????????????
        else:
            for i in range(-1, -input_dim - 1, -1):
                if x.shape[i] != out_grad.shape[i]:
                    broadcast_dim.append(i)
            for i in range(-input_dim - 1, -output_dim - 1, -1):
                broadcast_dim.append(i)
        return out_grad.sum(axes=tuple(broadcast_dim)).reshape(x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        input_shape = x.shape
        new_shape = list(input_shape)
        if self.axes is not None:
            for a in self.axes:
                new_shape[a] = 1
        else:
            return out_grad.broadcast_to(input_shape)

        return out_grad.reshape(new_shape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lfs, rhs = node.inputs
        # X @ Y????????????X????????????Y
        lfs_grad = out_grad @ array_api.transpose(rhs)

        rhs_grad = array_api.transpose(lfs) @ out_grad
        # print("??????????????????:",lfs_grad.shape)
        # print("??????????????????:",rhs_grad.shape)
        # print("????????????",lfs.shape)
        # print("????????????",rhs.shape)
        # print("?????????",tuple(range(len(rhs_grad.shape)-len(rhs.shape))))
        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????(?????????????????????????????????)???
        if lfs.shape != lfs_grad.shape:
            lfs_grad = lfs_grad.sum(axes=tuple(range(len(lfs_grad.shape) - len(lfs.shape))))
        if rhs.shape != rhs_grad.shape:
            rhs_grad = rhs_grad.sum(axes=tuple(range(len(rhs_grad.shape) - len(rhs.shape))))
        return lfs_grad, rhs_grad

    ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a*-1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad / x
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad * (x.realize_cached_data() > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        #axes???????????????????????????axes??????????????????????????????LogSumExp??????
        #??????axes????????????????????????????????????????????????????????????

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z,self.axes,keepdims=1)
        #?????????Z_max[0] broadcast??? Z[0][...][...]
        res = array_api.log(array_api.exp(Z-Z_max).sum(self.axes,keepdims=1))+Z_max

        new_shape=[]
        #??????????????????reshape(*shape),??????shape?????????????????????axes????????????
        if self.axes!=None:
            for i in range(len(Z.shape)):
                if i not in self.axes:
                    new_shape.append(Z.shape[i])
        return res.reshape(tuple(new_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        Z = node.inputs[0]

        if self.axes !=None:
            new_shape = list(Z.shape)
            for i in self.axes:
                new_shape[i]=1
            new_node = node.reshape(tuple(new_shape))
            new_grad = out_grad.reshape(tuple(new_shape))
        else:
            new_node = node
            new_grad = out_grad

        # print(node.shape, Z.shape, node_new.shape, out_grad.shape)
        #???????????????exp(Z-LogSumExp(Z))
        return new_grad*exp(Z-new_node)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
