import numpy as np
from naiveb.minimize import Minimize

class Cluster:

  '''
  This class is not working properly (IN DEVELOPMENT)

  -----------------------------------------------------------------------------
  attributes:

    num: number of clusters in the model
    dim: dimension of the num parameters

    like: the likelihood conditional to a parameter
    part: 1st partial derivatives of the likelihood
    hess: 2nd partial derivatives of the likelihood
    prior: probabilities of parameters in the model
    theta: the num guessed parameters for the model

  methods:

    __init__:
    __add_:
    log_like:
    grad_log:
    inv_hess:
    calibrator:
    
  '''


  def __init__(self, num, dim, like, part, hess, prior, theta):
    self.num = num
    self.dim = dim
    self.like = like
    self.part = part
    self.hess = hess
    self.prior = prior
    self.theta = theta


  def __add__(self, X):
    likelihoods = [np.array([self.like(x, theta) for theta in list(self.theta)]) * self.prior for x in X]
    return np.array([x / x.sum() for x in likelihoods]).mean(axis = 0)
    

  def log_like(self, X):
    if self.num == 1:
      temp = np.array([np.log(self.like(x, self.theta)) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      temp = np.array([np.log(x.sum()) for x in st]).sum(axis = 0)
    return temp


  def grad_log(self, X):
    if self.num == 1:
      temp = np.array([self.part(x, self.theta) / self.like(x, self.theta) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      nd = [np.array([self.part(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      temp = np.array([y / x.sum() for x, y in zip(st, nd)]).sum(axis = 0)
    if self.dim != 1:
      temp = np.concatenate(list(temp))
    return temp


  def inv_hess(self, X):
    if self.num == 1:
      temp = np.array([self.hess(x, self.theta) / self.like(x, self.theta) - np.tensordot( self.part(x, self.theta) / self.like(x, self.theta), 0 ) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      nd = [np.array([self.part(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      rd = [np.array([self.hess(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      temp = np.array([z / x.sum() - np.tensordot(y, y, 0) / x.sum()**2 for x, y, z in zip(st, nd, rd)]).sum(axis = 0)
      if self.dim == 1:
        temp = np.diag(temp)
      else:
        temp = [[np.zeros(self.dim,self.dim * k), temp[k], np.zeros(self.dim,self.dim * (self.num - 1 - k))] for k in range(self.num)]
        temp = np.block(temp)
    return np.linalg.inv(temp)
  

  def calibrator(self, X):

    def func(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      self.prior = self + X
      func = self.log_like(X)
      self.theta, self.prior = temp
      return - func

    def grad(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      self.prior = self + X
      grad = self.grad_log(X)
      self.theta, self.prior = temp
      return - grad
    
    def hess(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      self.prior = self + X
      hess = self.inv_hess(X)
      self.theta, self.prior = temp
      return - hess
    
    def update(guess):
      print('good work')
      self.theta = guess.reshape([self.num, self.dim])
      self.prior = self + X

    return Minimize(self.dim * self.num, func, grad = grad, hess = hess, guess = np.concatenate(list(self.theta)), update = update)