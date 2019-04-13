from typing import Dict, Optional, Tuple, Type

import copy
import numpy as np

class LineSearchMethod(object):
    def __init__(
        self,
        sigma: float,
        tau: Optional[float] = None,
        beta: Optional[float] = None,
        s_armijo: Optional[float] = None,
        step_size_rule: str = 'armijos',
    ):
        assert sigma < 1 and sigma > 0
        self.sigma = sigma

        if tau is not None:
            assert tau > sigma and tau < 1
            self.tau = tau

        if beta is not None:
            assert beta > 0 and beta < 1
            self.beta = beta

        if s_armijo is not None:
            assert s_armijo > 0
            self.s_armijo = s_armijo

        assert step_size_rule in ['goldstein', 'wp', 'strong-wp', 'armijos']
        if step_size_rule != 'armijos':
            raise NotImplementedError("Only Armijo's rule supported.")
        self.step_size_rule = step_size_rule

    @staticmethod
    def phi_fn(iterate, fn, descent_dir):
        phi = lambda alpha: fn.evaluate(iterate + alpha * descent_dir) \
            - fn.evaluate(iterate)
        grad_phi_alpha = lambda alpha: np.dot(
            fn.gradient(iterate + alpha * descent_dir), descent_dir)
        return phi, grad_phi_alpha

    def line_search(self, fn, iterate, descent_dir, grad_fx):
        phi, grad_phi_alpha = self.phi_fn(
            iterate=iterate, fn=fn, descent_dir=descent_dir)

        if self.step_size_rule == 'armijos':
            MAX_ARMIJO_TRIALS = 1000
            trial_idx = 0
            assert self.s_armijo is not None and self.beta is not None,\
                "Need to set s_armijo and beta for armijo's rule!"

            alpha = self.s_armijo
            while(trial_idx < MAX_ARMIJO_TRIALS):
                # print("Trying alpha: {}".format(alpha))
                if self.sigma_condition(alpha, phi, grad_phi_alpha):
                    # print("Succeeded sigma condition on alpha: {}".format(alpha))
                    return alpha
                else:
                    alpha = alpha * self.beta
                trial_idx += 1

            raise RuntimeError("Max armijo trials exceeded!")
        else:
            raise NotImplementedError


    def sigma_condition(self, alpha, phi, grad_phi_alpha) -> bool:
        return phi(alpha) <= self.sigma * alpha * grad_phi_alpha(0)

    def goldstein_condition(self, alpha, phi, grad_phi_alpha) -> bool:
        return phi(alpha) >= self.tau * alpha * grad_phi_alpha(0)

    def wp_condition(self, alpha, phi, grad_phi_alpha) -> bool:
        return grad_phi_alpha(alpha) >= self.tau * grad_phi_alpha(0)

    def strong_wp_condition(self, alpha, phi, grad_phi_alpha) -> bool:
        return np.abs(grad_phi_alpha(alpha)) <= -self.tau * grad_phi_alpha(0)
