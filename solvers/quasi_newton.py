from typing import Dict, Optional, Tuple

import copy
import numpy as np

from .base_solver import BaseSolver

from line_search import LineSearchMethod

def is_nonzerofinite(arr):
    # Reference: github.com/BRML/climin/blob/master/climin/base.py#L88
    """Return True if the array is neither zero, NaN or infinite."""
    return (arr != 0).any() and np.isfinite(arr).all()


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
        self.curr_direction = -1. * np.matmul(self.curr_H, self.grad_fx)

    def rank_two_update(self, grad_fx_prev, p_k, q_k):
        raise NotImplementedError("Child class must implement rank_two_update")

    def step(self):
        self.iter += 1
        alpha = self.alpha

        if not is_nonzerofinite(self.curr_direction):
            raise RuntimeError("Descent direction is not non-zero finite!")

        if self.use_line_search:
            alpha = self.ls_method.line_search(
                fn=self.fn,
                iterate=self.curr_iterate,
                descent_dir=self.curr_direction,
                grad_fx=self.grad_fx,
                min_step_size=1e-10,
            )

        # Compute next iterate: x_{k+1}
        next_iterate = self.curr_iterate + alpha * self.curr_direction

        # Caching g_k from self.grad_fx (before it gets overwritten by update)
        grad_fx_prev = copy.copy(self.grad_fx)

        prev_iterate = copy.copy(self.curr_iterate)

        # This will compute g_{k+1} and update self.grad_fx
        # This will also update self.fx
        self.update(next_iterate)

        p_k = self.curr_iterate - prev_iterate
        q_k = self.grad_fx - grad_fx_prev

        self.curr_H = self.rank_two_update(self.curr_H, grad_fx_prev, p_k, q_k)

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

    def __str__(self):
        return "dfp"

    def rank_two_update(self, prev_H, grad_fx_prev, p_k, q_k):
        pq_inner = np.dot(p_k, q_k)
        pp_outer = np.outer(p_k, p_k)
        H_q_mul = np.matmul(prev_H, q_k)
        q_H_q_mul = np.dot(q_k, H_q_mul)

        H_new = prev_H \
            + pp_outer / pq_inner \
            - np.outer(H_q_mul, H_q_mul) / q_H_q_mul
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

    def __str__(self):
        return "bfgs"

    def rank_two_update(self, prev_H, grad_fx_prev, p_k, q_k):
        pp_outer = np.outer(p_k, p_k)
        pq_inner = np.dot(p_k, q_k)
        H_q_mul = np.matmul(prev_H, q_k)
        q_H_q_mul = np.dot(q_k, H_q_mul)
        p_H_mul = np.dot(p_k, prev_H)

        term_1 = 1 + (q_H_q_mul / pq_inner)
        term_1 *= (pp_outer / pq_inner)

        term_2 = np.outer(p_k, p_H_mul)
        term_2 += np.outer(H_q_mul, p_k)
        term_2 /= pq_inner

        H_new = prev_H + term_1 - term_2
        return H_new


class LBFGSSolver(BaseSolver):
    def __init__(
        self,
        fn,
        x0,
        initial_hessian_diag: float = 1.,
        queue_size: int = 4,
        alpha: Optional[float] = None,
        iter: int = 0,
        term_crit: str = 'fn',
        use_line_search: bool = False,
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
        assert queue_size >= 1
        self.queue_size = queue_size
        # self.curr_H = np.eye(x0.shape[0])
        self.curr_H_diag = initial_hessian_diag
        self.curr_direction = -1. * self.curr_H_diag * self.grad_fx
        self.p_k_queue = np.zeros((self.queue_size, *self.grad_fx.shape))
        self.q_k_queue = np.zeros((self.queue_size, *self.grad_fx.shape))
        self.que_ids = []

    def __str__(self):
        return "l-bfgs"

    def step(self):
        self.iter += 1
        alpha = self.alpha

        if not is_nonzerofinite(self.curr_direction):
            raise RuntimeError("Descent direction is not non-zero finite!")

        if self.use_line_search:
            alpha = self.ls_method.line_search(
                fn=self.fn,
                iterate=self.curr_iterate,
                descent_dir=self.curr_direction,
                grad_fx=self.grad_fx,
                min_step_size=1e-8,
            )

        # Compute next iterate: x_{k+1}
        next_iterate = self.curr_iterate + alpha * self.curr_direction

        # Caching g_k from self.grad_fx (before it gets overwritten by update)
        grad_fx_prev = copy.copy(self.grad_fx)

        # self.prev_direction = copy.copy(self.curr_direction)
        self.prev_iterate = copy.copy(self.curr_iterate)

        # This will compute g_{k+1} and update self.grad_fx
        # This will also update self.fx
        self.update(next_iterate)

        p_k = self.curr_iterate - self.prev_iterate
        q_k = self.grad_fx - grad_fx_prev

        p_q_dot = np.dot(p_k, q_k)

        if p_q_dot > 1e-10:
            if not self.que_ids:
                # queues are empty initially
                curr_que_id = 0
            elif len(self.que_ids) < self.queue_size:
                # queues are partially full
                curr_que_id = self.que_ids[-1] + 1
            else:
                # queues are full
                curr_que_id = self.que_ids.pop(0)

            self.q_k_queue[curr_que_id] = q_k
            self.p_k_queue[curr_que_id] = p_k
            self.que_ids.append(curr_que_id)
            self.curr_H_diag = p_q_dot / np.dot(q_k, q_k)
        else:
            print("WARN: p_q_dot is too small or negative!")
            if p_q_dot < 0:
                raise RuntimeError("Descent direction is not gradient"
                " related, hessian approximation might not be psd.")

        self.curr_direction = self.find_next_direction()

    def find_next_direction(self):
        curr_que_size = len(self.que_ids)
        u_dir = -self.grad_fx

        alphas = np.zeros((curr_que_size))
        rhos = np.zeros((curr_que_size))

        for i in self.que_ids:
            rhos[i] = 1 / np.dot(self.q_k_queue[i], self.p_k_queue[i])

        for i in self.que_ids[::-1]:
            # Backward computation
            alphas[i] = np.dot(self.p_k_queue[i], u_dir) * rhos[i]
            u_dir -= alphas[i] * self.q_k_queue[i]

        r_direction = self.curr_H_diag * u_dir

        for i in self.que_ids:
            # Forward computation
            beta = rhos[i] * np.dot(self.q_k_queue[i], r_direction)
            r_direction += self.p_k_queue[i] * (alphas[i] - beta)

        return r_direction
