import numpy as np
from naiveb.minimize import Minimize

def normalize(x): return x/x.sum()

class Cluster:


  def __init__(self, num, dim, like, part, hess, prior, theta):
    self.num = num
    self.dim = dim
    self.like = like
    self.part = part
    self.hess = hess
    self.prior = prior
    self.theta = theta


  def posterior(self, x):
    likelihoods = np.array([self.like(x, theta) for theta in list(self.theta)]) * self.prior
    return likelihoods / likelihoods.sum()


  def __call__(self, X):
    return np.array([self.posterior(x) for x in list(X)])


  def __add__(self, X):
    self.prior = self(X).mean()


  def log_like(self, X):
    return np.array([np.array([self.like(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ]).sum()


  def gradient(self, X):
    likelihoods = np.array([np.array([self.like(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ])
    lik_par_der = np.array([np.array([self.part(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ])
    return (lik_par_der / likelihoods).sum()


  def inv_hess(self, X):
    likelihoods = np.array([np.array([self.like(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ])
    lik_par_der = np.array([np.array([self.part(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ])
    lik_sec_der = np.array([np.array([self.hess(x, theta) for theta in list(self.theta)]) @ self.prior for x in X ])
    log_sec_der = (lik_sec_der / likelihoods - np.tensordot(lik_par_der / likelihoods, lik_par_der / likelihoods, 0)).sum()
    return np.linalg.inv(log_sec_der)
  

  def calibrator(self, X):

    def func(theta):
      self.theta = theta
      self + X
      return self.log_like(X)

    def grad(theta):
      self.theta = theta
      self + X
      return self.gradient(X)
    
    def hess(theta):
      self.theta = theta
      self + X
      return self.inv_hess(X)
    
    return Minimize(self.dim * self.num, func, grad = grad, hess = hess, guess = self.theta)
    