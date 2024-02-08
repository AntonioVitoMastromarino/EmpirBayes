import queue
import numpy as np

class Linear:

  '''
  Class to approximate a linear map learning the inputs with rank 1 corrections
  -----------------------------------------------------------------------------

    attributes:

      dim: dimensions of the inputs of the map.
      inputs: the list of all the given inputs.
      output: the list of all the given output.
      matrix: when possible rapresents the map.

    methods:

      __init__: initialize the attributes.
      __call__: approximates map on input.
      __add__: updates inputs and outputs.
  
  '''


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
    if self.rec < self.dim:
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
      y /= np.linalg.norm(x)
      x /= np.linalg.norm(x)
      self.matrix += np.tensordot(y - self(x), x)