import queue
import numpy as np

class Linear:


  def __init__(self, dim):
    self.dim=dim
    self.rec=0
    self.inputs=[]
    self.output=[]
  
  
  def __call__(self, x):
    if self.rec < self.dim:
      x_dep=np.array([(x @ u)*u for u in self.inputs])
      y_dep=np.array([(x @ u)*v for (u,v) in zip(self.inputs, self.output)])
      return x-x_dep,y_dep
    else:
      return self.matrix @ x
  
  
  def __add__(self, xy):
    x, y = xy
    if self.rec<self.dim:
      x_dep = np.array([(x @ u) * u for u in self.inputs]).sum()
      y_dep = np.array([(x @ u) * v for (u, v) in zip(self.inputs, self.output)]).sum()
      x_dep = x - x_dep
      y_dep = y - y_dep
      norm = np.linalg.norm(x_dep)
      if (norm != 0):
        self.inputs.append(x_dep / norm)
        self.output.append(y_dep / norm)
        self.rec += 1
        if (self.rec == self.dim):
          self.matrix = np.array([np.tensordot(v, u) for (u, v) in zip(self.inputs, self.output)]).sum()
    else:
      self.matrix+=np.tensordot(y - self(x), x)