import ipdb
import argparse
import numpy as np

from functions import RosenbrockFn, PowellFn
from solvers import (SteepestDescentSolver,
                    ConjugateGradientSolver,
                    DFPSolver,
                    BFGSSolver)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--function',
        type=str,
        choices=['rosenbrock', 'powell'],
        default='rosenbrock'
    )
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--solver',
        type=str,
        choices=['steepest-descent', 'conjugate-gradient', 'dfp', 'bfgs'],
        default='bfgs'
    )

    parser.add_argument('--cg_variant',
        type=str,
        choices=['fr', 'pr'],
        default='fr'
    )

    parser.add_argument('--line_search_method',
        type=str,
        choices=['armijos', 'constant'],
        default='armijos',
    )

    parser.add_argument('--armijos_s', type=float, default=1.0)
    parser.add_argument('--armijos_beta', type=float, default=0.5)
    parser.add_argument('--ls_sigma', type=float, default=0.1)

    parser.add_argument('--term_crit',
        type=str,
        choices=['fn', 'grad'],
        default='grad'
    )
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=1)

    args = parser.parse_args()

    '''
        Init function
    '''
    if args.function == 'rosenbrock':
        fn = RosenbrockFn(args.n)
    elif args.function == 'powell':
        fn = PowellFn(args.n)

    '''
        Init solver
    '''
    ls_method_kwargs = dict(
        sigma=args.ls_sigma,
        tau=None,
        beta=args.armijos_beta,
        s_armijo=args.armijos_s,
        step_size_rule=args.line_search_method,
    )

    if args.solver == 'steepest-descent':
        solver = SteepestDescentSolver(
            fn=fn,
            x0=np.asarray([-1.2, 1, -1.2, 1], dtype=np.float32),
            alpha=args.alpha,
            term_crit=args.term_crit,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = ls_method_kwargs,
        )

    elif args.solver == 'conjugate-gradient':
        solver = ConjugateGradientSolver(
            fn=fn,
            x0=np.asarray([3, -1, 0, 1], dtype=np.float32),
            alpha=args.alpha,
            term_crit=args.term_crit,
            variant=args.cg_variant,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = ls_method_kwargs,
        )

    elif args.solver == 'dfp':
        solver = DFPSolver(
            fn=fn,
            x0=np.asarray([3, -1, 0, 1], dtype=np.float32),
            alpha=args.alpha,
            term_crit=args.term_crit,
            # variant=args.cg_variant,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = ls_method_kwargs,
        )

    elif args.solver == 'bfgs':
        solver = BFGSSolver(
            fn=fn,
            x0=np.asarray([3, -1, 0, 1], dtype=np.float32),
            alpha=args.alpha,
            term_crit=args.term_crit,
            # variant=args.cg_variant,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = ls_method_kwargs,
        )

    else:
        raise ValueError("Unrecognized solver: {}".format(args.solver))

    '''
        Iterate
    '''
    print ("iter: {0}, fx={1:.3f}".format(solver.iter, solver.fx))
    while (not solver.termination_criteria_reached()):
        solver.step()

        if solver.iter % args.log_every == 0:
            print ("iter: {0}, fx={1:.6f}, ||grad_fx||={2:.6f}".format(
                solver.iter, solver.fx, solver.grad_fx_norm
            ))

        if solver.iter == args.max_iters: break
