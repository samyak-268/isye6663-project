import copy
import numpy as np

from .base_solver import BaseSolver

from line_search import LineSearchMethod

class QuasiNewtonMethod(BaseSolver):
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
        self.curr_H = np.eye(x0.shape[0])
        self.prev_H = None
        self.prev_iterate = None
        self.curr_direction = -1. * np.matmul(self.curr_H, self.grad_fx)
        self.prev_direction = None

    def rank_two_update(self, grad_fx_prev, p_k, q_k):
        raise NotImplementedError("Child class must implement rank_two_update")

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

        self.prev_direction = copy.copy(self.curr_direction)
        self.prev_H = copy.copy(self.curr_H)
        self.prev_iterate = copy.copy(self.curr_iterate)

        # This will compute g_{k+1} and update self.grad_fx
        # This will also update self.fx
        self.update(next_iterate)

        p_k = self.curr_iterate - self.prev_iterate
        q_k = self.grad_fx - grad_fx_prev

        H_new = self.rank_two_update(grad_fx_prev, p_k, q_k)
        self.curr_H = H_new

        self.curr_direction = -1. * np.matmul(self.curr_H, self.grad_fx)


class DFPSolver(QuasiNewtonMethod):
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

    def rank_two_update(self, grad_fx_prev, p_k, q_k):
        H_q_mul = np.matmul(self.prev_H, q_k)
        _div_1 = np.dot(p_k, q_k)
        _div_2 = np.dot(q_k, H_q_mul)

        H_new = self.prev_H \
            + np.outer(p_k, p_k) / _div_1 \
            + np.outer(H_q_mul, H_q_mul) / _div_2
        return H_new


class BFGSSolver(QuasiNewtonMethod):
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

    def rank_two_update(self, grad_fx_prev, p_k, q_k):
        H_q_mul = np.matmul(self.prev_H, q_k)
        q_H_q_mul = np.dot(q_k, H_q_mul)
        pp_outer = np.outer(p_k, p_k)
        pq_inner = np.dot(p_k, q_k)

        term_1 = 1 + (q_H_q_mul / pq_inner)
        term_1 *= (pp_outer / pq_inner)

        term_2 = np.matmul(pp_outer, self.prev_H)
        term_2 += np.outer(H_q_mul, p_k)
        term_2 /= pq_inner

        H_new = self.prev_H + term_1 - term_2
        return H_new
