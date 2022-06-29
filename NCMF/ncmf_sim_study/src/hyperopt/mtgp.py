import numpy as np
import GPy
import GPyOpt
import sys
import time

class MTGPModel(GPyOpt.models.base.BOModel):

    # SET THIS LINE TO True of False DEPENDING IN THE ANALYTICAL GRADIENTS OF THE PREDICTIONS ARE AVAILABLE OR NOT

    analytical_gradient_prediction = False

    def __init__(self,max_iters=10,W=None,kappa=None):
        self.model = None
        self.optimizer = 'bfgs'
        self.max_iters = max_iters
        self.verbose = True
        self.optimize_restarts = 1
        self.W = W
        self.kappa = kappa


    def _create_model(self, X, Y):
        st = time.time()
        """
        Creates the model given some input data X and Y.
        """
        print("self.X.shape: ",X.shape)
        print("self.Y.shape: ",Y.shape)
        self.X = X
        self.Y = Y
        self.input_dim = X.shape[1]
        self.num_op = Y.shape[1]
        K = GPy.kern.Matern32(self.input_dim)
        if self.W is None:
            self.W = np.atleast_2d(np.array(np.ones(self.num_op))).T 
        if self.kappa is None:
            self.kappa = np.array(np.ones(self.num_op))
        icm = GPy.util.multioutput.ICM(input_dim=self.input_dim,num_outputs=self.num_op,kernel=K,W=self.W,W_rank=self.W.shape[1],kappa=self.kappa)
        X_list = []
        Y_list = []
        for i in np.arange(Y.shape[1]):
           #print("i: ",i)
           X_list.append(X.copy())
           Y_list.append(np.atleast_2d(Y[:,i]).T)
           #print("X_list[i].shape: ",X_list[i].shape)
           #print("Y_list[i].shape: ",Y_list[i].shape)
           #print("---")
        #print("creating model..")
        #print("len(X_list): ",len(X_list))
        #print("len(Y_list) - : ",len(Y_list))
        #self.model = GPy.models.GPCoregionalizedRegression([X],[Y],kernel=icm)
        self.model = GPy.models.GPCoregionalizedRegression(X_list,Y_list,kernel=icm)
        #self.model = GPy.models.GPRegression(X,Y,kernel=K)
        #print("Model creation done. ")
        et = time.time()
        #print("_create_model took ",(et-st)/60.0," secs.")
        #print("#")

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        st = time.time()
        """
        Updates the model with new observations.
        """
        if X_new is not None:
            print("Updating X,Y data")
            print("X_all.shape: ",X_all.shape)
            print("X_new.shape: ",X_new.shape)
            X_all = np.vstack([X_all, X_new])
            print("Y_all.shape: ",Y_all.shape)
            print("Y_new.shape: ",Y_new.shape)
            Y_all = np.vstack([Y_all, Y_new])

        print("updateModel: ")
        print("X_all.shape: ",X_all.shape)
        print("Y_all.shape: ",Y_all.shape)
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)
        et = time.time()
        #print("updateModel took ",(et-st)/60.0," secs.")
        #print("#")

    def _predict(self, X, full_cov, include_likelihood):
        # if X.ndim == 1:
        #     X = X[None,:]
        newX_all_tasks = None
        if X.shape[1] == self.input_dim:
            newX_list = []
            for i in np.arange(self.num_op):
                newX = np.atleast_2d(X)
                newX = np.hstack([newX,np.atleast_2d(np.ones(newX.shape[0])).T * i])
                newX_list.append(newX)    
            newX_all_tasks = np.vstack(newX_list)
        else:
            newX_all_tasks = X
        noise_dict = {'output_index':np.atleast_2d(newX_all_tasks[:,newX_all_tasks.shape[1]-1]).T.astype(int)}
        #print("debug##:")
        #print("X.shape: ",X.shape)
        #print("noise_dict: ",noise_dict)
        try:
            if X.shape[0] > 100:
                raise Exception("inner")
        except:
            tb = sys.exc_info()[2]
            raise Exception("outer").with_traceback(tb)
        #print("newX_all_tasks.shape: ",newX_all_tasks.shape)
        #print("newX_all_tasks: ",newX_all_tasks)
        m, v = self.model.predict(newX_all_tasks, full_cov=full_cov, include_likelihood=include_likelihood,Y_metadata=noise_dict)
        num_all = m.shape[0]
        num_per_task = int(num_all/self.num_op)
        m_temp = m[0:num_per_task]
        v_temp = v[0:num_per_task]
        temp_start_idx = num_per_task
        #print("#Computing norm:")
        #print("debug - start")
        #print("m.shape: ",m.shape)
        #print("v.shape: ",v.shape)
        #print("self.num_op: ",self.num_op)
        #print("num_per_task: ",num_per_task)
        for op in np.arange(2,self.num_op+1):
            temp_end_idx = op*num_per_task
            #print("temp_start_idx: ",temp_start_idx," temp_end_idx: ",temp_end_idx)
            m_temp+= m[temp_start_idx:temp_end_idx]
            v_temp+= v[temp_start_idx:temp_end_idx]
            temp_start_idx = temp_end_idx
        #print("m_temp.shape: ",m_temp)
        #print("v_temp.shape: ",v_temp)
        #print("#debug - end")
        m = m_temp
        v = v_temp
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        st = time.time()
        m, v = self._predict(X, False, with_noise)  
        # We can take the square root because v is just a diagonal matrix of variances
        et = time.time()
        #print("Predict took ",(et-st)/60.0," secs.")
        #print("#")
        return m, np.sqrt(v)

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        noise_dict = {'output_index':np.atleast_2d(self.model.X[:,self.model.X.shape[1]-1]).T.astype(int)}
        return self.model.predict(self.model.X,Y_metadata=noise_dict)[0].min()

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])