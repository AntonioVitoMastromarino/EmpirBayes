from typing import Any
import numpy as np
from naiveb.minimize import Minimize

class Cluster:

  '''
  This class shall use the methods in Minimize to search for maximum likelihood
  for a statistical model where the distribution is a mixture of a fixed number
  of conditional distributions. This class is not working properly: DEVELOPMENT
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

    __init__: initializes the attributes
    __add_: adjust weight in the mixture
    log_like: compute the log likelihood
    grad_log: compute the log derivative
    inv_hess: compute an inverse hessian
    calibrator: return a Minimize object
    __call__: search for a max-estimator

  '''

  def __init__(self, num, dim, like, part, hess, prior, theta, constrain = lambda x: True):

    '''
    initializes the attributes

    Inputs:
    Output:

    '''

    self.gap = None
    self.num = num
    self.dim = dim
    self.like = like
    self.part = part
    self.hess = hess
    self.prior = prior
    self.theta = theta
    self.constrain = constrain


  def __add__(self, X):

    '''
    adjust weight in the mixture

    Inputs:

      list of samples

    Output:

      observed weight 

    '''

    likelihoods = [np.array([self.like(x, theta) for theta in list(self.theta)]) * self.prior for x in X]
    return np.array([x / x.sum() for x in likelihoods]).mean(axis = 0)
    

  def log_like(self, X):

    '''
    compute the log likelihood

    Inputs:

      list of samples

    Output:

      a loglikelihood

    '''

    if (self.num == 1):
      for x in X:
        assert self.like(x, self.theta) > 0
      temp = np.array([np.log(self.like(x, self.theta)) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]).sum() for x in X]
      for x in st:
        assert x > 0
      temp = np.log(np.array(st)).sum(axis = 0)      
    return temp


  def grad_log(self, X):

    '''
    compute the log derivative

    Inputs:

      list of samples

    Output:

      gradient of log

    '''

    if (self.num == 1):
      return np.array([self.part(x, self.theta) / self.like(x, self.theta) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      nd = [np.array([self.part(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      temp = np.array([b / a.sum() for a, b in zip(st, nd)]).sum(axis = 0)
      if (self.dim == 1):
        return temp
      else:
        return np.concatenate(list(temp))
    

  def inv_hess(self, X):

    '''
    compute an inverse hessian

    Inputs:

      list of samples

    Output:

      inverse hessian

    '''

    if (self.num == 1):
      temp = np.array([self.hess(x, self.theta) / self.like(x, self.theta) - np.tensordot(self.part(x, self.theta) / self.like(x, self.theta), 0 ) for x in X ]).sum(axis = 0)
    else:
      st = [np.array([self.like(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      nd = [np.array([self.part(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      rd = [np.array([self.hess(x, list(self.theta)[k]) * self.prior[k] for k in range(self.num)]) for x in X]
      if (self.dim == 1):
        temp = [np.diag(list(z)) for z in rd]
        rd = temp
      else:
        temp = [np.concatenate(list(y)) for y in nd]
        nd = temp
        temp = [np.block([[np.zeros([self.dim, self.dim * k]), list(z)[k], np.zeros([self.dim, self.dim * (self.num - 1 - k)])] for k in range(self.num)]) for z in rd]
        rd = temp
      temp = np.array([c / a.sum() - np.tensordot(b, b, 0) / a.sum()**2 for a, b, c in zip(st, nd, rd)]).sum(axis = 0)    
    return np.linalg.inv(temp)
  

  def calibrator(self, X):

    '''
    return a Minimize object

    Inputs:

      list of samples

    Output:

      Minimize object

    '''

    def func(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      func = self.log_like(X)
      self.theta, self.prior = temp
      return - func

    def grad(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      grad = self.grad_log(X)
      self.theta, self.prior = temp
      return - grad
    
    def hess(guess):
      temp = self.theta, self.prior
      self.theta = guess.reshape([self.num, self.dim])
      hess = self.inv_hess(X)
      self.theta, self.prior = temp
      return - hess
    
    def update(guess):
      self.theta = guess.reshape([self.num, self.dim])
      posterior = self + X
      self.gap = np.linalg.norm(self.prior - posterior)
      self.prior = posterior

    def constrain(guess):
      temp = [self.constrain(t) for t in list(guess.reshape([self.num, self.dim]))]
      return np.array(temp).all()

    return Minimize(self.dim * self.num, func, grad = grad, hess = hess, guess = np.concatenate(list(self.theta)), constrain = constrain, update = update)
  

  def __call__(self, X, **args):

    '''
    search for a max-estimator:

    Inputs:

      rate_rd: the initial variance to be used in stochastic descent
      rate_gd: the initial learning rate to use for gradiant descent
      iter_rd: initial number of iterations to be used in stochastic
      iter_gd: initial number of iterations used in gradient descent
      iter_nt: initial number of iterations used in Newoton's method
      toll_theta: tollerance for the dual gap of the theta parameter
      toll_prior: tollerance for the dual gap of the omega parameter

    Output:

      prior: the omega parameter optimized by the minimizer protocol 
      theta: the theta parameter optimized by the minimizer protocol 

    '''

    call = self.calibrator(X)
    call.protocol([args['rate_rd'], args['rategd']],
                  [args['iter_rd'], args['iter_gd'], args['iter_nt']],
                  args['toll_theta'],
                  condition = lambda: self.gap < args['toll_prior'])
    return self.prior, self.theta