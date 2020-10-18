import numpy as np
import jax.numpy as jnp
from scipy.linalg import sqrtm

from . import mcmc_helper as helper

__all__ = ["sampler"]


class HMCSampler(object):
    """
    A class for doing Hamiltonian Monte-Carlo

    :param ndim: Dimension of the parameter space being sampled
    :param psi: The negative of the log-posterior, a.k.a Potential
    :param grad_psi: The gradient of the Potential
    :param Hamiltonian_mass: array of masses for the hamil
    :param psi_args: (optional) extra positional arguments for the function psi
    :param grad_psi_args: (optional) extra positional arguments for the function grad_psi
    """

    def __init__(self, ndim, psi, grad_psi, Hamiltonian_mass, verbose=False, psi_args=[], grad_psi_args=[]):
        self.ndim = ndim
        self.psi = psi
        self.grad_psi = grad_psi
        self.Hamiltonian_mass = Hamiltonian_mass
        self.psi_args = psi_args
        self.grad_psi_args = grad_psi_args
        self.psi_kwargs = None
        self.grad_psi_kwargs = None
        self.verbose = verbose

    def sample(self, init_pos, time_step, N_ITERATIONS=5000, N_BURN_IN=100, N_LEAPFROG=20):
        """
        Samples the parameter space 'N_ITERATIONS' times
        :param init_pos: The initial guess from where the chains are started
        :param time_step: Time step for the step through parameter space.
        :param N_ITERATIONS: Number of interations to be advanced
        :param N_LEAPFROG: Number of leapfrog steps
        """
        self._chain = np.zeros((N_ITERATIONS, self.ndim))
        self._posterior = np.zeros(N_ITERATIONS)
        self._chain[0, :] = init_pos
        self.n_accepted = 0
        self.iterations = N_ITERATIONS

        x_old = init_pos
        self._posterior[0] = -self.get_psi(init_pos)

        mean = x_old

        M_cov = self.Hamiltonian_mass
        cov = 1.0/M_cov

        for i in range(N_ITERATIONS):
            x_old, lnprob, accepted = self.sample_one_step(x_old, time_step, N_LEAPFROG)

            if(accepted):
                self.n_accepted += 1
            self._chain[i, :] = x_old
            self._posterior[i] = lnprob

    def sample_one_step(self, x_old, time_step, N_LEAPFROG, higher_order_leapfrog=False, psi_kwargs=None, grad_psi_kwargs=None):
        """
        A function to run one step of HMC
        """
        if(psi_kwargs is not None):
            self.psi_kwargs=psi_kwargs
        if(grad_psi_kwargs is not None):
            self.grad_psi_kwargs=grad_psi_kwargs
        p = np.random.normal(np.zeros(self.ndim), np.sqrt(self.Hamiltonian_mass))

        phi = np.random.uniform(0., 2 * np.pi, size=self.ndim)
        J = np.complex(0,1)
        complex_p = np.cos(phi) + np.sin(phi) * J
        p = p * complex_p
        p = jnp.array(p)
        H_old = self.Hamiltonian(x_old, p)

        dt = np.random.uniform(0, time_step)
        N = np.random.randint(1,N_LEAPFROG)

        if(self.verbose):
            print("dt: %2.3f, N:%d"%(dt, N))

        x_proposed, p_proposed = self.leapfrog(x_old, p, dt, N)

        H_proposed = self.Hamiltonian(x_proposed, p_proposed)

        diff = H_proposed-H_old

        print("diff: %2.3f"%(diff))
        accepted = False
        if(diff < 0.0):
            x_old = x_proposed
            accepted = True
        else:
            rand = np.random.uniform(0.0, 1.0)
            log_rand = np.log(rand)

            if(-log_rand > diff):
                x_old = x_proposed
                accepted = True

        return x_old, -self.get_psi(x_old), accepted

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
    def acceptance_fraction(self):
        """
        The fraction of proposed steps that were accepted.

        """
        return self.n_accepted / self.iterations

    def leapfrog(self, x, p, dt, N):
        """
        Returns the new position and momenta evolved from the position {x,p} in phase space, with `N_LEAPFROG` leapfrog iterations.
        :param x: The current position in the phase space
        :param p: The current momenta
        :param dt: The time step to which it is evolved
        :param N_LEAPFROG: Number of leapfrog iterations
        """
        x_old = x
        p_old = p

        for i in range(N):
            psi_grad = self.get_grad_psi(x_old)

            p_new = p_old-(dt/2)*psi_grad
            x_new = x_old + dt*(p_new/self.Hamiltonian_mass)

            psi_grad = self.get_grad_psi(x_new)
            p_new = p_new-(dt/2)*psi_grad

            x_old, p_old = x_new, p_new

        return x_new, p_new

    def Hamiltonian(self, x, p):
        """
        Returns the hamiltonian for a given position and momenta
        :param x: The position in the parameter space
        :param p: The set of momenta
        """
        M = self.Hamiltonian_mass
        H = np.sum(0.5* np.abs(p)**2 /M) + self.get_psi(x)
        return H

    def get_psi(self, x):
        """Return psi at the given position."""
        if(self.psi_kwargs is not None):
            return self.psi(x, *self.psi_args, **self.psi_kwargs)
        return self.psi(x, *self.psi_args)

    def get_grad_psi(self, x):
        """Return grad_psi at the given position."""
        if(self.grad_psi_kwargs is not None):
            return self.grad_psi(x, *self.grad_psi_args, **self.grad_psi_kwargs)
        return self.grad_psi(x, *self.grad_psi_args)

    def num_grad_psi(self, x, delta_x):
        """
        Returns numerical gradient of psi
        """
        x_new = np.zeros(x.shape).astype(np.complex)

        for i in range(len(x)):
            x_new[i] = x[i]

        num_grad_psi_real = np.zeros(len(x))
        num_grad_psi_imag = np.zeros(len(x))

        L0 = self.get_psi(x)

        for i in range(len(x)):
            if(i%100 == 0):
                print(i)
            eps = delta_x[i]
            x_new[i] += np.complex(eps, 0)
            L1_real = self.get_psi(x_new)
            num_grad_psi_real[i] = (L1_real - L0) / eps
            x_new[i] = x[i]

            x_new[i] += np.complex(0, eps)
            L1_imag = self.get_psi(x_new)
            num_grad_psi_imag[i] = (L1_imag - L0) / eps
            x_new[i] =  x[i]

        return [num_grad_psi_real, num_grad_psi_imag]
