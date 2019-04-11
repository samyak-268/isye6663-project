import numpy as np

class RosenbrockSubFn():
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, x):
        i = (self.idx - 1)
        if self.idx % 2 == 0:
            return 1 - x[i-1]
        else:
            return 10 * (x[i+1] - np.square(x[i]))

    def gradient(self, x):
        grad = np.zeros(x.shape)

        i = (self.idx - 1)
        if self.idx % 2 == 0:
            grad[i-1] = -1
        else:
            grad[i] = (-20. * x[i])
            grad[i+1] = 10.

        return grad

class RosenbrockFn():
    def __init__(self, n):
        assert n % 2 == 0, "n = {} is not a multiple of 2.".format(n)
        self.n = n
        self.fns = [ RosenbrockSubFn(i+1) for i in range(self.n) ]

    def evaluate(self, x):
        val = 0.
        for fn in self.fns:
            val += np.square(fn.evaluate(x))
        return val

    def gradient(self, x):
        grad = np.zeros(x.shape)
        for fn in self.fns:
            fx = fn.evaluate(x)
            grad_fx = fn.gradient(x)
            grad += (fx * grad_fx)
        return 2 * grad


if __name__=='__main__':
    rosenbrock = RosenbrockFn(4)
    x = np.asarray([-1.2, 1, -1.2, 1], dtype=np.float32)

    fx = rosenbrock.evaluate(x)
    grad_fx = rosenbrock.gradient(x)

    print (fx)
    print (grad_fx)
