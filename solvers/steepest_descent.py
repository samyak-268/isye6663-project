import copy
import numpy as np

class SteepestDescentSolver():
    def __init__(self, fn, x0, alpha, iter=0, term_crit='fn'):
        self.EPS = 1e-6
        self.fn = fn
        self.curr_iterate = x0
        self.alpha = alpha
        self.iter = iter
        self.term_criteria = term_crit

        self.update(self.curr_iterate)

        '''
            Caching because required for evaluating termination criteria
        '''
        self.f_x0 = copy.copy(self.fx)
        self.grad_f_x0_norm = copy.copy(self.grad_fx_norm)

    def update(self, x):
        self.fx = self.fn.evaluate(self.curr_iterate)
        self.grad_fx = self.fn.gradient(self.curr_iterate)
        self.grad_fx_norm = np.linalg.norm(self.grad_fx)

    def step(self):
        self.iter += 1
        next_iterate = self.curr_iterate - self.alpha * self.grad_fx

        self.curr_iterate = next_iterate
        self.update(self.curr_iterate)

    def termination_criteria_fn(self):
        if (self.fx / self.f_x0) <= self.EPS: return True
        return False

    def termination_criteria_grad(self):
        if self.grad_fx_norm <= self.EPS * (self.grad_f_x0_norm + 1.): return True
        return False

    def termination_criteria_reached(self):
        if self.term_criteria == 'fn':
            return self.termination_criteria_fn()
        else:
            return self.termination_criteria_grad()
