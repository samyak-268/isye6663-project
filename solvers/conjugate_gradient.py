import copy
import numpy as np

class ConjugateGradientSolver():
    def __init__(
        self,
        fn,
        x0,
        alpha,
        iter=0,
        term_crit='fn',
        variant='fr',
        use_line_search=False,
        ls_method_kwargs=None,
    ):
        self.EPS = 1e-6
        self.fn = fn
        self.alpha = alpha
        self.iter = iter
        self.term_criteria = term_crit
        self.variant = variant
        self.n = x0.shape[0]
        self.use_line_search = use_line_search

        if use_line_search:
            self.ls_method = LineSearchMethod(**ls_method_kwargs)
        else:
            assert self.alpha is not None
            self.ls_method = None

        self.update(x0)
        self.curr_direction = -1. * self.grad_fx

        '''
            Caching because required for evaluating termination criteria
        '''
        self.f_x0 = copy.copy(self.fx)
        self.grad_f_x0_norm = copy.copy(self.grad_fx_norm)

    def update(self, x):
        self.curr_iterate = x
        self.fx = self.fn.evaluate(self.curr_iterate)
        self.grad_fx = self.fn.gradient(self.curr_iterate)
        self.grad_fx_norm = np.linalg.norm(self.grad_fx)

    def compute_beta(self, prev_grad, curr_grad):
        curr_grad_norm = np.linalg.norm(curr_grad)
        prev_grad_norm = np.linalg.norm(prev_grad)

        if self.variant == 'fr':
            return np.square(curr_grad_norm / prev_grad_norm)
        elif self.variant == 'pr':
            return np.dot(curr_grad, (curr_grad - prev_grad)) / np.square(prev_grad_norm)

    def step(self):
        self.iter += 1
        alpha = self.alpha

        if self.use_line_search:
            alpha = self.ls_method.line_search(
                fn=self.fn,
                iterate=self.curr_iterate,
                descent_dir=self.curr_direction,
                grad_fx=self.grad_fx,
            )

        # Compute next iterate: x_{k+1}
        next_iterate = self.curr_iterate + alpha * self.curr_direction

        # Caching g_k from self.grad_fx (before it gets overwritten by update)
        grad_fx_prev = copy.copy(self.grad_fx)

        # This will compute g_{k+1} and update self.grad_fx
        # This will also update self.fx
        self.update(next_iterate)

        # Use g_k and g_{k+1} to compute beta (FR)
        if self.iter % self.n == 0:
            beta = 0
        else:
            beta = self.compute_beta(
                prev_grad=grad_fx_prev,
                curr_grad=self.grad_fx
            )

        # Compute next direction: d_{k+1} and update self.curr_direction
        next_direction = -1. * self.grad_fx + beta * self.curr_direction
        self.curr_direction = next_direction


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
