from sys import maxunicode
import numpy as np
from naiveb.linear import Linear

class Minimize:


  def __init__(self, dim, func, grad = None, hess = None, guess = 0):
  
    self.dim = dim
    self.func = func

    if grad is None:
      self.grad = Linear(dim)
      self.grad_available = False
    else:
      self.grad = grad
      self.grad_available = True    

    if hess is None:
      self.hess = Linear(dim)
      self.hess_available = False
    else:
      self.hess = hess
      self.hess_available = True

    self.guess = guess


  def n_step(self):

    if self.grad_available:
      gradient=self.grad(self.guess)
    else:
      try:
        gradient=self.grad.matrix
      except:
        rand = np.random.randn(self.dim)/self.dim
        x, y = self.grad(rand)
        gradient = y * (rand - x) + x

    if self.hess_available:
      newton_step = self.hess(self.guess)@gradient
    else:
      try:
        newton_step = self.hess(gradient)
      except:
        x, y = self.hess(gradient)
        newton_step = y + np.linalg.norm(x)*np.random.randn(self.dim)/self.dim

    if self.func(self.guess - newton_step) < old_func:
      old_func = self.func(self.guess)
      self.guess -= newton_step
      if not self.grad_available:
        self.grad+(newton_step,old_func-self.func(self.guess))
        if not self.hess_available:
          try:
            self.hess + (gradient - self.grad.matrix, newton_step)
          except:
            rand = np.random.randn(self.dim)/self.dim
            x, y = self.grad(rand)
            self.hess + (gradient - y*(rand-x) + np.linalg.norm(x)*np.random.randn(self.dim)/self.dim, newton_step)
      elif not self.hess_available:
        self.hess+(gradient-self.grad(self.guess), newton_step)
    else:
      raise Exception('Newton got stuck')


  def d_step(self, learn_rate, randn_rate):

    if self.grad_available:
      gradient=self.grad(self.guess)
    else:
      try:
        gradient=self.grad.matrix
      except:
        rand = np.random.randn(self.dim)/self.dim
        x, y = self.grad(rand)
        gradient = y * (rand - x) + x
    gra_descend = gradient * learn_rate + np.random.randn(self.dim)/self.dim * randn_rate

    if self.func(self.guess - gra_descend) < old_func:
      old_func = self.func(self.guess)
      self.guess -= gra_descend
      if not self.grad_available:
        self.grad+(gra_descend,old_func-self.func(self.guess - gra_descend))
        if not self.hess_available:
          try:
            self.hess + (gradient - self.grad.matrix, gra_descend)
          except:
            rand = np.random.randn(self.dim)/self.dim
            x, y = self.grad(rand)
            self.hess + (gradient - y*(rand-x) + np.linalg.norm(x)*np.random.randn(self.dim)/self.dim, gra_descend)
      elif not self.hess_available:
        self.hess+(gradient-self.grad(self.guess - gra_descend), gra_descend)
    else:
      raise Exception('Descend got stuck')


  def step(self, rate, max_succ, max_fail, max_iter):
    
    c = np.cos(np.pi/(2 * self.dim))
    s = np.sin(np.pi/(2 * self.dim))
    learn_rate = rate
    randn_rate = 0
    succ = 0
    fail = 0
    
    while (succ < max_succ and fail < max_fail and self.iter < max_iter):
      self.iter += 1
      try:
        self.d_step(learn_rate, randn_rate)
        if (randn_rate > 0):
          learn_rate = c * learn_rate + s * randn_rate
          randn_rate = c * randn_rate - s * learn_rate
        else:
          succ += 1
        fail = 0      
      except:
        if (learn_rate > 0):
          learn_rate = c * learn_rate - s * randn_rate
          randn_rate = c * randn_rate + s * learn_rate
        else:
          fail += 1
        succ = 0

    self.iter += 1
    try:
      old_func = self.func(self.guess)
      self.n_step()
      self.gap = old_func - self.func(self.guess)
    except:
      if (fail == max_fail):
        self.gap = 0
      elif (succ == max_succ):
        self.gap = None
      else:
        self.gap = None  
        return c
    
  def __call__(self, max_toll, max_iter):
    self.iter = 0
    rate = max_toll
    c = self.step(rate, 6, 6, 3 * self.dim)
    while(self.iter < max_iter):
      if self.gap is None:
        if c is None:
          rate = 2 * rate
        else:
          rate = c * rate
        c = self.step(rate, 6, 6, self.iter + 3 * self.dim)
      elif(self.gap < max_toll):
        return self.guess
      else:
        self.n_step()
        self.iter += 1
    return self.guess
