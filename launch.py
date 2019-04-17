import os
import copy
import shlex
import subprocess
from subprocess import Popen
import ipdb

class Launcher():

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

    def __init__(self, functions, dims, solvers, wait_for_all=True):
        self.functions = functions
        self.dims = dims
        self.solvers = solvers
        self.wait_for_all = wait_for_all

    def get_cmds_to_run(self):
        self.cmds = []
        for fn in self.functions:
            for n in self.dims:
                for solver in self.solvers:
                    args = copy.deepcopy(Launcher.default_args)
                    args['FUNCTION'] = fn
                    args['N'] = n
                    args['SOLVER'] = solver
                    if '-cg' in args['SOLVER']:
                        args['CG_VARIANT'] = args['SOLVER'].split('-')[0]
                        args['SOLVER'] = 'conjugate-gradient'

                    if 'lbfgs-' in args['SOLVER']:
                        args['lbfgs_queue_size'] = args['SOLVER'].split('-')[1]
                        args['SOLVER'] = 'lbfgs'

                    self.cmds.append(Launcher.cmd_template.format(**args))

    def launch_job(self, cmd):
        # print(cmd)
        # subprocess.check_call(shlex.split(cmd))
        return Popen(cmd, shell=True)

    def launch_all(self):
        launcher.get_cmds_to_run()
        for i, cmd in enumerate(self.cmds):
            print("Launching job {}...".format(i+1))
            proc = self.launch_job(cmd)
            if self.wait_for_all:
                print("Waiting for job {} to finish".format(i+1))
                proc.wait()


if __name__=='__main__':

    functions = ['rosenbrock', 'powell']
    dims = [4, 64, 1024, 2048]
    solvers = ['steepest-descent', 'fr-cg', 'pr-cg', 'bfgs', 'dfp',
        'lbfgs-2', 'lbfgs-4', 'lbfgs-8']
    # solvers = ['lbfgs-2', 'lbfgs-4', 'lbfgs-8']

    launcher = Launcher(functions, dims, solvers)
    launcher.launch_all()
