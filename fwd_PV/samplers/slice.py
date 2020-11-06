import numpy as np
from . import mcmc_helper as helper

class SliceSampler():
    """
    A class for doing Slice Sampling

    :param ndim: Dimension of the parameter space being sampled
    :param lnprob: The log-posterior.
    :param step_size: Initial Step size
    :param lnprob_args: (optional) extra positional arguments for the function lnprob
    """

    def __init__(self, ndim, lnprob, step_size_init, lnprob_args=[]):
        self.ndim = ndim
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args
        self.lnprob_kwargs = None
        self.step_size = step_size_init

    def sample(self, init_pos, N_ITERATIONS=1000, N_BURN_IN=100, N_ADAPT=500, step_update_parameter=1.0, verbose=False):
        """
        Samples the parameter space 'N_ITERATIONS' times
        :param init_pos: (1D Array of size ndim) The initial guess from where the chain is started
        :param step_size_init: (1D Array of size ndim) Initial guess for the size of a step-out
        :param N_ITERATIONS: Number of interations of sampling. Default is 1000.
        :param N_ADAPT: Number of initial step during which the step size is adapted. Default is 500
        :param step_update_parameter: A hyperparameter to decide the step size in terms of the standard deviation.
                                    Default is 1. Works best if it is O(1).
        """
        assert init_pos.shape[0] == self.ndim, "init_pos must be an 1-D array of size n_dim"
        assert step_size_init.shape[0] == self.ndim, "step_size_init must be an 1-D array of size n_dim"

        self._chain = np.zeros((N_ITERATIONS, self.ndim))
        self._posterior = np.zeros(N_ITERATIONS)
        self._chain[0, :] = init_pos
        self.num_iterations = N_ITERATIONS

        x = init_pos
       
        mu, cov = init_pos, self.step_size*self.step_size

        for i in range(N_ITERATIONS):
            if(verbose):
                if(i%100 == 0):
                    print(str(i)+"-th step in progress")
            for d in range(self.ndim):
                x = self.one_step_slice_sampler(x, d)
            if((i < (N_BURN_IN + N_ADAPT)) and (i >= N_BURN_IN)):
                mu, cov = helper.update_step_size(i + 2 - N_BURN_IN, x, mu, cov)
                self.step_size = step_update_parameter * np.sqrt(cov)
            self._chain[i] = x
            self._posterior[i] = self.get_lnprob(x)

    def sample_one_step(self, x, lnprob_kwargs=None):
        """
        """
        if(lnprob_kwargs is not None):
            self.lnprob_kwargs=lnprob_kwargs
        lnprob_x = self.get_lnprob(x)
        log_u = np.log(np.random.rand()) + lnprob_x
        x_left, x_right = self.create_interval(x, log_u)
        while(True):
            x_prime = x_left + (x_right - x_left)*np.random.rand()
            lnProb_prime = self.get_lnprob(x_prime)
            if(lnProb_prime > log_u):
                break
            else:
                x_left, x_right = self.modify_interval(x, x_prime, x_left, x_right)
        return x_prime

    def create_interval(self, x, log_u):
        """
        """
        r = np.random.rand()

        x_left = x - r*self.step_size
        x_right = x + (1.0-r)*self.step_size

        while(self.get_lnprob(x_left) > log_u):
            x_left = x_left - self.step_size
        while(self.get_lnprob(x_right) > log_u):
            x_right = x_right + self.step_size
        return x_left, x_right

    @staticmethod
    def modify_interval(x, x_prime, x_left, x_right):
        """
        """
        if(x_prime > x):
            x_right = x_prime
            assert x_right >= x, "Right direction "+str(x_right)+" must be greater than x, "+str(x)
        else:
            x_left = x_prime
            assert x_left <= x, "Left direction "+str(x_left)+" must be smaller than x, "+str(x)
        return x_left, x_right

    @property
    def chain(self):
        """
        Return the chain of the sampler.
        """
        return self._chain

    @property
    def posterior(self):
        """
        Return the chain of the sampler.
        """
        return self._posterior

    def get_lnprob(self, x):
        """Return lnprob at the given position."""
        return self.lnprob(x, *self.lnprob_args, **self.lnprob_kwargs)
