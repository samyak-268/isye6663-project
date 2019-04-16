import os
import pickle
import ipdb
import argparse
import numpy as np

from functions import RosenbrockFn, PowellFn
from solvers import SteepestDescentSolver, ConjugateGradientSolver
from utils import Logger, get_output_fname

def get_initial_iterate(args):
    x0 = np.zeros((args.n,), dtype=np.float32)

    if args.function == 'rosenbrock':
        even_idxs = np.arange(0, x0.shape[0], 2)
        odd_idxs = np.arange(1, x0.shape[0], 2)
        x0[even_idxs], x0[odd_idxs] = -1.2, 1.
    elif args.function == 'powell':
        first_set_idxs = np.arange(0, x0.shape[0], 4)
        second_set_idxs = np.arange(1, x0.shape[0], 4)
        third_set_idxs = np.arange(2, x0.shape[0], 4)
        fourth_set_idxs = np.arange(3, x0.shape[0], 4)

        x0[first_set_idxs] = 3.
        x0[second_set_idxs] = -1.
        x0[third_set_idxs] = 0.
        x0[fourth_set_idxs] = 1.

    return x0

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
        choices=['steepest-descent', 'conjugate-gradient'],
        default='steepest-descent'
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

    parser.add_argument('--armijos_s', type=float, default=0.5)
    parser.add_argument('--armijos_beta', type=float, default=0.5)
    parser.add_argument('--ls_sigma', type=float, default=0.1)

    parser.add_argument('--term_crit',
        type=str,
        choices=['fn', 'grad'],
        default='fn'
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
    if args.solver == 'steepest-descent':
        solver = SteepestDescentSolver(
            fn=fn,
            x0=get_initial_iterate(args),
            alpha=args.alpha,
            term_crit=args.term_crit,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = dict(
                sigma=args.ls_sigma,
                tau=None,
                beta=args.armijos_beta,
                s_armijo=args.armijos_s,
                step_size_rule=args.line_search_method,
            )
        )
    elif args.solver == 'conjugate-gradient':
        solver = ConjugateGradientSolver(
            fn=fn,
            x0=get_initial_iterate(args),
            alpha=args.alpha,
            term_crit=args.term_crit,
            variant=args.cg_variant,
            use_line_search=args.line_search_method != 'constant',
            ls_method_kwargs = dict(
                sigma=args.ls_sigma,
                tau=None,
                beta=args.armijos_beta,
                s_armijo=args.armijos_s,
                step_size_rule=args.line_search_method,
            )
        )

    '''
        Iterate
    '''
    logger = Logger(solver)

    logger.save()
    logger.log()
    while (not solver.termination_criteria_reached()):
        solver.step()
        logger.save()
        if solver.iter % args.log_every == 0:
            logger.log()
        if solver.iter == args.max_iters: break


    '''
        Save result
    '''
    output_fname = get_output_fname(args, solver)
    output_fpath = os.path.join('notebooks/data', output_fname)
    output = {'args': args, 'plot_data': logger.data}
    with open(output_fpath, 'wb') as f: pickle.dump(output, f)
