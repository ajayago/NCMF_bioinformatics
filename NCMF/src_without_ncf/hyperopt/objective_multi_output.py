import numpy as np
import GPy
import GPyOpt
import time

class SingleObjectiveMultiOutput(GPyOpt.core.task.objective.Objective):

    def __init__(self, func, num_cores = 1, objective_name = 'no_name', batch_type = 'synchronous', space = None):
        self.func  = func
        self.n_procs = num_cores
        self.num_evaluations = 0
        self.space = space
        self.objective_name = objective_name

    def evaluate(self, x):
        #print("SingleObjectiveMultiOutput: ")
        #print("x: ",x)
        #print("x.shape: ",x.shape)
        f_evals, cost_evals = self._eval_func(x)
        return f_evals, cost_evals

    def _eval_func(self, x):
        cost_evals = []
        f_evals  = [] #np.empty(shape=[0, 1])
        for i in range(x.shape[0]):
            #st_time    = time.time()
            rlt,cur_cost = self.func(np.array([x[i]])) #return 1xd
            f_evals.append(rlt) #     = np.vstack([f_evals,rlt])
            #cost_evals += [time.time()-st_time]
            cost_evals += [cur_cost]
        return np.concatenate(f_evals), cost_evals