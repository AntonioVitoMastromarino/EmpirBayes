import numpy as np
from naiveb.minimize import Minimize

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
    return likelihoods / likelihoods.sum(axis = 0)
    #invalid value encountered in divide! Likelihoods are too concentrated!!!


  def __call__(self, X):
    return np.array([self.posterior(x) for x in list(X)])


  def __add__(self, X):
    self.prior = self(X).mean(axis = 0)


  def log_like(self, X):
    if self.num == 1:
      temp = np.array([np.log(self.like(x, self.theta)) for x in X ]).sum(axis = 0)
    else:
      temp = [np.array([np.log(self.like(x, theta)) for theta in list(self.theta)]) @ self.prior for x in X]
      temp = np.array(temp).sum(axis = 0)
    return temp

  def grad_log(self, X):
    if self.num == 1:
      temp = np.array([self.part(x, self.theta) / self.like(x, self.theta) for x in X ]).sum(axis = 0)
    else:
      temp = [np.array([self.part(x, theta) / self.like(x, theta) for theta in list(self.theta)]) * self.prior for x in X]
      temp = np.array(temp).sum(axis = 0)
    if self.dim != 1:
      temp = np.concatenate(list(temp))
    return temp
      
  def inv_hess(self, X):
    if self.num == 1:
      temp = np.array([self.hess(x, self.theta) / self.like(x, self.theta) - np.tensordot( self.part(x, self.theta) / self.like(x, self.theta), 0 ) for x in X ]).sum(axis = 0)
    else:
      temp = [np.array([self.hess(x, theta) / self.like(x, theta) - np.tensordot( self.part(x, theta) / self.like(x, theta), 0 ) for theta in list(self.theta)]) * self.prior for x in X]
      temp = np.array(temp).sum(axis = 0)
      if self.dim == 1:
        temp = np.diag(temp)
      else:
        temp = [[np.zeros(self.dim,self.dim * k), temp[k], np.zeros(self.dim,self.dim * (self.num - 1 - k))] for k in range(self.num)]
        temp = np.block(temp)
    return np.linalg.inv(temp)
  

  def calibrator(self, X):

    def func(theta):
      self.theta = theta
      self + X
      return self.log_like(X)

    def grad(theta):
      self.theta = theta
      self + X
      return self.grad_log(X)
    
    def hess(theta):
      self.theta = theta
      self + X
      return self.inv_hess(X)
    
    def update(guess):
      self.theta = guess.reshape([self.num, self.dim])

    temp = Minimize(self.dim * self.num, func, grad = grad, hess = hess, guess = np.concatenate(list(self.theta)), update = update)