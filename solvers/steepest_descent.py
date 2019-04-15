import copy
import numpy as np

from .base_solver import BaseSolver

from line_search import LineSearchMethod

class SteepestDescentSolver(BaseSolver):
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
        super().__init__(
            fn=fn,
            x0=x0,
            alpha=alpha,
            iter=iter,
            term_crit=term_crit,
            use_line_search=use_line_search,
            ls_method_kwargs=ls_method_kwargs,
        )
        pass

    def step(self):
        self.iter += 1
        alpha = self.alpha

        if self.use_line_search:
            alpha = self.ls_method.line_search(
                fn=self.fn,
                iterate=self.curr_iterate,
                descent_dir= -1 * self.grad_fx,
                grad_fx=self.grad_fx,
            )

        next_iterate = self.curr_iterate - alpha * self.grad_fx

        self.curr_iterate = next_iterate
        self.update(self.curr_iterate)
