

def get_output_fname(args, solver):
    path = "fn={0}_n={1}_solver={2}".format(
        args.function,
        args.n,
        str(solver)
    )
    armijos_params = "{0}+{1}+{2}".format(
        args.armijos_s,
        args.armijos_beta,
        args.ls_sigma
    )
    path += "_lr={}_alpha={}_armijosParams={}".format(
        args.line_search_method,
        args.alpha,
        armijos_params
    )
    path += "_termCrit={}_maxIters={}.pkl".format(
        args.term_crit,
        args.max_iters
    )
    return path


class Logger():
    def __init__(self, solver, tb_writer=None):
        self.solver = solver
        self.tb_writer = tb_writer
        self.data = []

    def save(self):
        self.data.append({
            'iter': self.solver.iter,
            'fx': self.solver.fx,
            'grad_fx': self.solver.grad_fx
        })

    def graph(self):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('fn_value', self.solver.fx, self.solver.iter)
            self.tb_writer.add_scalar('grad_norm', self.solver.grad_fx_norm, self.solver.iter)

    def log(self):
        print ("iter: {0}, fx={1:.6f}, grad_fx_norm={2:.6f}".format(
            self.solver.iter, self.solver.fx, self.solver.grad_fx_norm
        ))
