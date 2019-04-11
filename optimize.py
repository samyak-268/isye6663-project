import ipdb
import argparse
import numpy as np

from functions import RosenbrockFn, PowellFn
from solvers import SteepestDescentSolver

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
        choices=['steepest_descent'],
        default='steepest_descent'
    )
    parser.add_argument('--term_crit',
        type=str,
        choices=['fn', 'grad'],
        default='fn'
    )
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=100)

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
    if args.solver == 'steepest_descent':
        solver = SteepestDescentSolver(
            fn=fn,
            x0=np.asarray([-1.2, 1, -1.2, 1], dtype=np.float32),
            alpha=args.alpha,
            term_crit=args.term_crit
        )

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
