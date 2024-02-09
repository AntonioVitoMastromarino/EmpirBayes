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
      constrain: ensure well defined.
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

      protocol:
        uses __call__ recursively adjusting params
        it accepts stopping condition from an user

  '''

  def __init__(self, dim, func, grad, hess, guess = 0, constrain = lambda t: True, update = None):

    '''
    initialize the attributes

    Inputs:

      dim: dimension of the argument.
      func: the function to minimize.
      grad: the gradient of function.
      hess: the inverse hessian func.
      guess: the initial guessed min.
      constrain: ensure well defined.
      update: see in method __call__.

    Output:

      object of the class *.Minimize.
      by default self.guess set to 0.
      constrain default returns True.
      the default for update is None.
      grad_avail: True if grad known.
      hess_avail: True if hess known.

    '''

    assert constrain(guess)

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
    self.constrain = constrain
    self.update = update

  def compute(self):

    '''
    compute gradient in guess
    eventually approximate it

    Inputs:

      the method takes no input
    
    Output:

      returns gradient in guess
      eventually approximate it

    '''

    if self.grad_avail:
      return self.grad(self.guess)
    else:
      try:
        return self.grad.matrix
      except:
        rand = np.random.randn(self.dim)/self.dim
        x, y = self.grad(rand)
        return y * (rand - x) + x


  def attempt(self, step, called = None):

    '''
    check new guesses comparing with the old
    if the attempt accepted update the guess
    update grad unless grad_avail holds true
    update hess unless hess_avail holds true

    Inputs:

      step

    Output:

      None

    '''

    if not self.constrain(self.guess - step):
      raise Warning(called + 'proposed a step that out of the prescribed domain. \n Current guess: ' + str(self.guess) + '\n Proposal: ' + str(self.guess - step))

    old_func = self.func(self.guess)
    old_grad = self.compute()
    if self.func(self.guess - step) < old_func:
      self.guess -= step
      if not self.grad_avail:
        self.grad + ( step, old_func - self.func(self.guess))
      if not self.hess_avail:
        self.hess + (old_grad - self.compute(), step)
    else:
      # here you can learn the gradient and the hessian when not available
      raise Exception('Got stuck')


  def nt_step(self):

    '''
    compute a newton-tangent guess
    eventually approximate hessian
    calls both compute and attempt

    Inputs:

      None
    
    Output:

      None

    '''
  
    gradient = self.compute()

    if self.hess_avail:
      step = self.hess(self.guess) @ gradient
    else:
      try:
        step = self.hess(gradient)
      except:
        x, y = self.hess(gradient)
        step = y + np.linalg.norm(x) * np.random.randn(self.dim) / self.dim

    try:
      self.attempt(step, called = "Newton's method")
    except:
      self.attempt(- step, called = "Newton's method")
      raise Warning("Backward direction accepted as a proposal in Newton's method.\n This may indicate proximity to a local maximum or to a saddle.")


  def gd_step(self, learn_rate, randn_rate):

    '''
    perform a gradient-descend
    calls compute for the grad
    calls attempt for checking

    Inputs:

      learn_rate, randn_rate

    Output:

      method gives no output

    '''

    step = self.compute() * learn_rate + np.random.randn(self.dim) / self.dim * randn_rate

    if randn_rate == 0:
      called = 'gd'
    elif learn_rate == 0:
      called = 'rd'
    else:
      called = 'st'

    try:
      self.attempt(step, called = called)
    except:
      self.attempt(- step, called = called)
      if called == 'gd':
        raise Warning("Backward direction accepted as a proposal in gradient descent.\n This may indicate that learning rate is too high (" + str(learn_rate) + ").")


  def __call__(self, rates, steps, condition = lambda: False):

    '''
    try to use in sequence gd_step and nt_step
    calls update for each iteration when given

    Inputs:

      rates as a list of 2 elements
      steps as a list of 3 elements
      condition as a bool procedure

    Output:

      succ: is a list of 3 integers
      fail: is a list of 3 integers

    '''

    succ = [0, 0, 0]
    fail = [0, 0, 0]

    rate = rates[0]
    iter = 0
    while (iter < steps[0]) and not condition(self):
      iter += 1
      try:
        self.gd_step(0, rate)
        rate += rates[0] / (steps[0] + 1)
        succ[0] += 1
      except:
        fail[0] +=1

    rate = rates[1]
    iter = 0
    while (iter < steps[1]) and not condition(self):
      iter += 1
      try:
        self.gd_step(rate, 0)
        succ[1] += 1
      except:
        rate -= rates[1] / (steps[1] + 1)
        fail[1] +=1

    iter = 0
    while (iter < steps[2]) and not condition(self):
      iter += 1
      try:
        self.nt_step()
        succ[2] += 1
      except:
        iter = steps[2]
        fail[2] +=1

    if (condition(self) and self.update is not None):
      self.update(self.guess)
    return succ, fail
  
  def protocol(self, rates, steps, toll, condition = lambda: False):

    '''
    uses __call__ recursively adjusting params
    it accepts stopping condition from an user

    Inputs:

      rates as a list of 2 elements
      steps as a list of 3 elements
      toll as a tollerance for grad
      condition as a bool procedure

    Output:

      the method do not give output

    '''

    C = np.array(3)
    def rot(x0, x1, c):
      if (x0 < 1) or (x1 < 1):
        return (x0, x1)
      y0 = np.cos(1 / 2**c) * x0 - np.sin(1 / 2**c) * x1
      y1 = np.cos(1 / 2**c) * x1 + np.sin(1 / 2**c) * x0
      if (y0 < 0) or (y1 < 0):
        c += 1
        return rot(x0, x1, c)
      else:
        c -= 1
        return (y0, y1)
    
    while (np.linalg.norm(self.grad(self.guess)) < toll) and not condition():

      succ, fail = self(rates, steps, condition = lambda: np.linalg.norm(self.grad(self.guess)) < toll)
      if (succ[0] > 0):
        rates[0] *= 1 + succ[0] / (steps[0] + 1)
      if (succ[1] > 0):
        rates[1] *= (2 * succ[1] + 1) / (steps[1] + 1)

      if (fail[2] > 0):
        steps[0], steps[1] = rot(steps[0], steps[1], C)
      else:
        steps[1], steps[0] = rot(steps[1], steps[0], C)
        rates[0] /= 2**steps[0]
    