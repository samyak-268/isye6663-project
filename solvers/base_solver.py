import copy
import numpy as np

import copy
import numpy as np

from line_search import LineSearchMethod

class BaseSolver(object):
    def __init__(
        self,
        fn,
        x0,
        alpha=None,
        iter=0,
        term_crit='fn',
        use_line_search=False,
        ls_method_kwargs=None,
    ):
        self.EPS = 1e-6
        self.fn = fn
        self.alpha = alpha
        self.iter = iter
        self.term_criteria = term_crit
        self.use_line_search = use_line_search

        if use_line_search:
            self.ls_method = LineSearchMethod(**ls_method_kwargs)
        else:
            assert self.alpha is not None
            self.ls_method = None

        self.update(x0)

        '''
            Caching because required for evaluating termination criteria
        '''
        self.f_x0 = copy.copy(self.fx)
        self.grad_f_x0_norm = copy.copy(self.grad_fx_norm)

    def update(self, x):
        self.curr_iterate = x
        self.fx = self.fn.evaluate(x)
        self.grad_fx = self.fn.gradient(x)
        self.grad_fx_norm = np.linalg.norm(self.grad_fx)

    def __str__(self):
        raise NotImplementedError("Child class must implement __str__ function!")

    def step(self):
        raise NotImplementedError("Child class must implement step function!")

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
