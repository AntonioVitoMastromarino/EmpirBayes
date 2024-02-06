from audioop import ratecv
from sys import maxunicode
import numpy as np
from naiveb.linear import Linear

class Minimize:

  '''
  Class to minimize a function using Newton Tangent or Gradient Descend method.
  -----------------------------------------------------------------------------
    attributes:

      dim: dimension of the argument.
      guess: the initial guessed min.
      func: the function to minimize.
      grad: the gradient of function.
      hess: the inverse hessian func.
      update: see in method __call__.
      grad_avail: True if grad known.
      hess_avail: True if hess known.
      
    methods:

      __init__:
        initialize the attributes

      compute:
        compute gradient in guess
        eventually approximate it

      attempt:
        check new guesses comparing with the old
        if the attempt accepted update the guess
        update grad unless grad_avail holds true
        update hess unless hess_avail holds true

      nt_step:
        compute a newton-tangent guess
        eventually approximate hessian
        calls both compute and attempt

      gd_step:
        perform a gradient-descend
        calls compute for the grad
        calls attempt for checking

      __call__:
        try to use in sequence gd_step and nt_step
        calls update for each iteration when given

  '''


  def __init__(self, dim, func, grad = None, hess = None, guess = 0, update = None):
  
    self.x_gap = None
    self.y_gap = None

    self.dim = dim
    self.func = func

    if grad is None:
      self.grad = Linear(dim)
      self.grad_avail = False
    else:
      self.grad = grad
      self.grad_avail = True    

    if hess is None:
      self.hess = Linear(dim)
      self.hess_avail = False
    else:
      self.hess = hess
      self.hess_avail = True

    self.guess = guess
    self.update = update


  def compute(self):

    if self.grad_avail:
      return self.grad(self.guess)
    else:
      try:
        return self.grad.matrix
      except:
        rand = np.random.randn(self.dim)/self.dim
        x, y = self.grad(rand)
        return y * (rand - x) + x


  def attempt(self, step):

    old_func = self.func(self.guess)
    old_grad = self.compute()
    if self.func(self.guess + step) < old_func:
      self.guess += step
      if not self.grad_avail:
        self.grad + (step, old_func - self.func(self.guess))
      if not self.hess_avail:
        self.hess + (old_grad - self.compute(), step)
    else:
      # coming soon: here you can learn the gradient and the hessian when not available
      raise Exception('Got stuck')


  def nt_step(self):

    gradient = self.compute()

    if self.hess_avail:
      step = self.hess(self.guess)@gradient
    else:
      try:
        step = self.hess(gradient)
      except:
        x, y = self.hess(gradient)
        step = y + np.linalg.norm(x)*np.random.randn(self.dim)/self.dim

    self.attempt(step)


  def gd_step(self, learn_rate, randn_rate):

    self.attempt(self.compute() * learn_rate + np.random.randn(self.dim)/self.dim * randn_rate)


  def __call__(self, rates, steps, toll, max_iter):

    rate0, rate1 = rates

    for k in range(max_iter):

      rate = rate0
      iter = 0
      while (iter < steps[0]):
        iter += 1
        try:
          self.d_step(0, rate)
        except:
          rate -= rate0 / (steps[0] + 1)
      rate0 = rate * (steps[0] + 1)

      rate = rate1
      iter = 0
      while (iter < steps[1]):
        iter += 1
        try:
          self.d_step(rate, 0)
        except:
          rate -= rate1 / (steps[1] + 1)
      rate1 = rate

      iter = 0
      while (np.linalg.norm(self.grad(self.guess)) > toll * (max_iter - k) and iter < steps[2] * k):
        iter += 1
        try:
          self.n_step()
        except:
          iter = steps[2]

      if (np.linalg.norm(self.grad(self.guess)) < toll):
        self.update(self.guess)