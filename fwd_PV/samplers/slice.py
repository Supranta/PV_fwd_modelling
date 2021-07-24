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

    def __init__(self, ndim, lnprob, lnprob_args=[], verbose=False):
        self.ndim = ndim
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args
        self.lnprob_kwargs = None
        self.verbose       = verbose
        self.mu            = None

    #===================================        
    def sample_one_step(self, x, lnprob_kwargs=None):
        if(self.mu is None):
            self.mu = x
        if(lnprob_kwargs is not None):
            self.lnprob_kwargs=lnprob_kwargs
        x_transformed = helper.coordinate_transform(x, self.eig_vecs)
        for d in range(self.ndim):
            if(self.verbose):
                print("Sampling dimension: %d"%(d))
            x_transformed, ln_prob = self.one_step_slice_sampler(x_transformed, d)
        x = helper.inverse_transform(x_transformed, self.eig_vecs)        
        acc = 1
        return x, acc, ln_prob
    
    def set_cov(self, cov):
        self.cov = cov
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov)
        self.step_size = np.sqrt(self.eig_vals)
        
    #===================================
    def one_step_slice_sampler(self, x_transformed, direction):
        """
        """
        lnprob_x = self.get_lnprob(helper.inverse_transform(x_transformed, self.eig_vecs))
        log_u = np.log(np.random.rand()) + lnprob_x
        if(self.verbose):
            print("Creating interval..")
        x_left, x_right = self.create_interval(x_transformed, direction, log_u)
        while(True):
            x_prime = x_left + (x_right - x_left)*np.random.rand()
            lnProb_prime = self.get_lnprob(helper.inverse_transform(x_prime, self.eig_vecs))
            if(lnProb_prime > log_u):
                break
            else:
                if(self.verbose):
                    print("Modifying interval...")
                lnprob_left = self.get_lnprob(helper.inverse_transform(x_left, self.eig_vecs))
                lnprob_right = self.get_lnprob(helper.inverse_transform(x_right, self.eig_vecs))
                x_left, x_right = self.modify_interval(x_transformed, x_prime, x_left, x_right, direction)
        return x_prime, lnProb_prime

    def create_interval(self, x_transformed, d, log_u):
        """
        """
        r = np.random.rand()

        x_left = helper.add_array(x_transformed, d, -r*self.step_size[d])
        x_right = helper.add_array(x_transformed, d, (1.0-r)*self.step_size[d])

        assert x_right[d] > x_left[d]
        while(self.get_lnprob(helper.inverse_transform(x_left,self.eig_vecs)) > log_u):
            x_left = helper.add_array(x_left, d, -self.step_size[d])
        while(self.get_lnprob(helper.inverse_transform(x_right,self.eig_vecs)) > log_u):
            x_right = helper.add_array(x_right, d, self.step_size[d])

        return x_left, x_right

    def update_cov(self, x, n):
        self.mu, self.cov = helper.update_step_size(n, x, self.mu, self.cov)
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov)
        self.step_size = np.sqrt(self.eig_vals)

    @staticmethod
    def modify_interval(x, x_prime, x_left, x_right, direction):
        """
        """
        if(x_prime[direction] > x[direction]):
            x_right = x_prime
            assert x_right[direction] >= x[direction], "Right direction "+str(x_right[direction])+" must be greater than x, "+str(x[direction])
        else:
            x_left = x_prime
            assert x_left[direction] <= x[direction], "Left direction "+str(x_left[direction])+" must be smaller than x, "+str(x[direction])
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

    @property
    def theta_cov(self):
        return self.cov

    def get_lnprob(self, x):
        """Return lnprob at the given position."""
        if(self.lnprob_kwargs is not None):
            return self.lnprob(x, *self.lnprob_args, **self.lnprob_kwargs)
        return self.lnprob(x, *self.lnprob_args)