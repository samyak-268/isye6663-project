import os
import copy
import shlex
import subprocess
import ipdb

default_args = dict(
    CG_VARIANT=None,
    ALPHA=0.0001,
    LINE_SEARCH_METHOD='armijos',
    ARMIJOS_S=0.5,
    ARMIJOS_BETA=0.5,
    LS_SIGMA=0.1,
    TERM_CRIT='fn',
    MAX_ITERS=10000
)

cmd_template = '''
python optimize.py \
    --function {FUNCTION} \
    --n {N} \
    --solver {SOLVER} \
    --cg_variant {CG_VARIANT} \
    --line_search_method {LINE_SEARCH_METHOD} \
    --alpha {ALPHA} \
    --armijos_s {ARMIJOS_S} \
    --armijos_beta {ARMIJOS_BETA} \
    --ls_sigma {LS_SIGMA} \
    --term_crit {TERM_CRIT} \
    --max_iters {MAX_ITERS} \
    --tb
'''


def launch_all(functions, dims, solvers):
    for fn in functions:
        for n in dims:
            for solver in solvers:
                args = copy.deepcopy(default_args)
                args['FUNCTION'] = fn
                args['N'] = n
                args['SOLVER'] = solver
                if '-cg' in args['SOLVER']:
                    args['CG_VARIANT'] = args['SOLVER'].split('-')[0]
                    args['SOLVER'] = 'conjugate-gradient'

                print (cmd_template.format(**args))
                subprocess.check_call(shlex.split(cmd_template.format(**args)))


if __name__=='__main__':

    functions = ['rosenbrock', 'powell']
    dims = [4, 64, 1024, 2048]
    solvers = ['steepest-descent']

    launch_all(functions, dims, solvers)
