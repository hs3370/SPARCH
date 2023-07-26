import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from arch import arch_model
from arch.univariate import ARX
import scipy
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize


from __future__ import annotations
from arch.univariate.distribution import Distribution
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Callable
import warnings

from numpy import (
    abs,    array,    asarray,    empty,    exp,    int64,    
    integer,    isscalar,    log,    nan,    ndarray, exp,
    ones_like,    pi,    sign,    sqrt,    sum, random)
from numpy.random import Generator, RandomState, default_rng
from scipy.special import comb, gamma, gammainc, gammaincc, gammaln
import scipy.stats as stats
from scipy.optimize import bisect

from arch.typing import ArrayLike, ArrayLike1D, Float64Array
from arch.utility.array import AbstractDocStringInheritor, ensure1d

class MixNormal(Distribution, metaclass=AbstractDocStringInheritor):
    """
    Mixture of two Normal distributions for use with SPARCH model

    Parameters
    ----------
    random_state : RandomState, optional
        .. deprecated:: 5.0

           random_state is deprecated. Use seed instead.

    seed : {int, Generator, RandomState}, optional
        Random number generator instance or int to use. Set to ensure
        reproducibility. If using an int, the argument is passed to
        ``np.random.default_rng``.  If not provided, ``default_rng``
        is used with system-provided entropy.
    """

    def __init__(
        self,
        random_state: RandomState | None = None,
        *,
        seed: None | int | RandomState | Generator = None,
    ) -> None:
        super().__init__(random_state=random_state, seed=seed)
        self._name = "Mixture of two Normal distributions"
        self.num_params: int = 3  

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return empty(0), empty(0)
        #return array([[1, 0, 0], [-1, 0, 0], [0, 1,0], [0, -1,0] , [0,0,1],[0,0,-1]]), array([0.05, 0.95, -20,20,0.2, 10])

    def bounds(self, resids: Float64Array) -> list[tuple[float, float]]:
        """
        Bounds of parameters:
        p1: (0,1)
        u1: (-10,10)
        sigma_1:(0.001,10)
        """
        return [(0.0001, 0.9999),(-10000,10000),(0.0001, 10000)]

    def loglikelihood(
        self,
        parameters: Sequence[float] | ArrayLike1D,
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> float | Float64Array:
        r"""Computes the log-likelihood of assuming residuals are mixture normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : ndarray
            Parameters of the first normal distribution: p1,u1,sigma1. Second one can be calculated by restrictions.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=
            \ln \frac{1}{\sqrt{h_t}} \left[
                 p_1  \frac{1}{\sqrt{ 2\pi\sigma_1^2} } exp\{ -\frac{(x-\mu_1)^2}{2\sigma_1^2} \}
                +(1-p_1)\frac{1}{\sqrt{ 2\pi\sigma_2^2} } exp\{ -\frac{(x-\mu_2)^2}{2\sigma_2^2} \} \right]

        """
        parameters = asarray(parameters, dtype=float)
        p1, u1, sigma_1_2 = parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        z = resids/sqrt(sigma2)
        warnings.filterwarnings("ignore")
        lls =-1/2 *log(sigma2) +  log(   p1/(sqrt(2*pi*sigma_1_2)) *  exp(-( z-u1)**2/(2*sigma_1_2)) 
                                     +   p2/(sqrt(2*pi*sigma_2_2)) *  exp(-( z-u2)**2/(2*sigma_2_2))   )
        warnings.filterwarnings("default")
          
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: Float64Array) -> Float64Array:
        """
        Starting values of parameters
        """
        #gmm = GaussianMixture(n_components=2).fit(std_resid.reshape(-1,1))
        #return array([gmm.weights_[0],gmm.means_[0][0], gmm.covariances_[0][0][0] ])
        return array([0.3, 0.1,1])
    
    def _simulator(self, size: int | tuple[int, ...]) -> Float64Array:
        assert self._parameters is not None
        p1, u1, sigma_1_2 = self._parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        Z = np.random.binomial(n=1, p=p1, size=size)
        return sqrt(sigma_1_2)**Z*sqrt(sigma_2_2)**(1-Z)*self._generator.standard_normal(size) + u1*Z+ u2*(1-Z)

    def simulate(
        self, parameters: int | float | Sequence[float | int] | ArrayLike1D
    ) -> Callable[[int | tuple[int, ...]], Float64Array]:
        parameters = ensure1d(parameters, "parameters", False)
        self._parameters = asarray(parameters, dtype=float)
        return self._simulator

    def parameter_names(self) -> list[str]:
        return ['p_1','mu_1','sigma_1^2']

    def cdf(
        self,
        resids: Sequence[float] | ArrayLike1D,
        parameters: None | Sequence[float] | ArrayLike1D = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        
        parameters = asarray(parameters, dtype=float)
        p1, u1, sigma_1_2 = parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        return p1*stats.norm.cdf(asarray((resids-u1)/sqrt(sigma_1_2))  ) + p2*stats.norm.cdf(asarray((resids-u2)/sqrt(sigma_2_2)))

    def ppf(
        self,
        pits: float | Sequence[float] | ArrayLike1D,
        parameters: None | Sequence[float] | ArrayLike1D = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        parameters = asarray(parameters, dtype=float)
        p1, u1, sigma_1_2 = parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        scalar = isscalar(pits)
        if scalar:
            pits = array([pits])
        else:
            pits = asarray(pits)
            
        def inverse_cdf(cdf, target_p, lower_bound=-100, upper_bound=100):
            def root_func(x):
                return cdf(x,parameters) - target_p
            return bisect(root_func, lower_bound, upper_bound)     
        
        ppf = inverse_cdf(self.cdf, pits, lower_bound=-100, upper_bound=100)

        if scalar:
            return ppf[0]
        else:
            return ppf

    def moment(
        self, n: int, parameters: None | Sequence[float] | ArrayLike1D = None
    ) -> float:
        if n < 0:
            return nan
        parameters = asarray(parameters, dtype=float)
        p1, u1, sigma_1_2 = parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        moment1 = stats.norm.moment(n,loc=u1,scale=sqrt(sigma_1_2))
        moment2 = stats.norm.moment(n,loc=u2,scale=sqrt(sigma_2_2))
        return p1 * moment1 + p2 * moment2

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: None | Sequence[float] | ArrayLike1D = None,
        num_samples=1000,
    ) -> float:
        
        parameters = asarray(parameters, dtype=float)
        p1, u1, sigma_1_2 = parameters
        p2 = 1- p1
        u2 = p1/(p1-1) *u1
        sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
        
        if n < 0:
            return nan
        elif n == 0:
            return cdf(z,parameters)
        elif n==1:
            return -p1*stats.norm.pdf(z,loc=u1,scale=sqrt(sigma_1_2))  -p2*stats.norm.pdf(z,loc=u2,scale=sqrt(sigma_2_2))
        else:
            -(z ** (n - 1)) * (p1*stats.norm.pdf(z,loc=u1,scale=sqrt(sigma_1_2))+p2*stats.norm.pdf(z,loc=u2,scale=sqrt(sigma_2_2))) 
            + (n - 1) * self.partial_moment(  n - 2, z, parameters  )

def print_param(params):
    """print distributional parameters"""
    p1,u1,sigma_1_2 = params
    p2 = 1- p1
    u2 = p1/(p1-1) *u1
    sigma_2_2 = (1-p2*u2*u2-p1*(sigma_1_2+u2*u2))/(p2)
    print('Mixture of 2 normals parameters: ')
    print("Probabilities: ", [p1,p2])
    print("Means: ", [u1,u2])
    print("Variances: ", [sigma_1_2,sigma_2_2])


def MofN_simulator(dist_params=[0.2,-0.6,0.7],size=5000):
    """
    Simulate SPARCH data with given distributional parameters
    
    Input:  dist_params[p1,u1,sigma1_2]
    return: DataFrame
        1.data: The simulated data, which includes any mean dynamics.
        2.volatility: The conditional volatility series
        3.errors: The simulated errors generated to produce the model. 
                    The errors are the difference between the data and its conditional mean, 
                    and can be transformed into the standardized errors by dividing by the volatility.
    """
    print('Simulating SPARCH data. Error follows: ')
    print_param(dist_params)
    gjr = arch_model(y=sample_data['Inflation shock'],mean='Zero', vol='GARCH',p=2, o=2,q=1).fit(disp='off')
    fake_params = list(gjr.params) + dist_params
    sim_sparch = arch_model(None, p=2, o=2, q=1, mean='Zero')
    sim_sparch.distribution = MixNormal()
    return sim_sparch.simulate(fake_params, size),fake_params

if __name__=='__main__':
    sim_data,fake_params = MofN_simulator()
    sparch = arch_model(sim_data.data, p=2, o=2, q=1, mean='Zero')
    sparch.distribution = MixNormal()
    sp = sparch.fit()#starting_values = fake_params)
    sp
