import numpy as np

class PowellSubFn():
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, x):
        i = (self.idx - 1)
        if self.idx % 4 == 1:
            return x[i] + (10. * x[i+1])
        elif self.idx % 4 == 2:
            return np.sqrt(5.0) * (x[i+1] - x[i+2])
        elif self.idx % 4 == 3:
            return np.square(x[i-1] - 2*x[i])
        else:
            return np.sqrt(10) * np.square(x[i-3] - x[i])

    def gradient(self, x):
        grad = np.zeros(x.shape)

        i = (self.idx - 1)
        if self.idx % 4 == 1:
            grad[i], grad[i+1] = 1., 10.
        elif self.idx % 4 == 2:
            grad[i+1], grad[i+2] = np.sqrt(5), -1. * np.sqrt(5)
        elif self.idx % 4 == 3:
            grad[i-1] = 2 * (x[i-1] - 2*x[i])
            grad[i] = -4 * (x[i-1] - 2*x[i])
        else:
            grad[i-3] = 2 * np.sqrt(10) * (x[i-3] - x[i])
            grad[i] = -2 * np.sqrt(10) * (x[i-3] - x[i])

        return grad

class PowellFn():
    def __init__(self, n):
        assert n % 4 == 0, "n = {} is not a multiple of 4.".format(n)
        self.n = n
        self.fns = [ PowellSubFn(i+1) for i in range(self.n) ]

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
    powell = PowellFn(4)
    x = np.asarray([3, -1, 0, 1], dtype=np.float32)

    fx = powell.evaluate(x)
    grad_fx = powell.gradient(x)

    print (fx)
    print (grad_fx)
