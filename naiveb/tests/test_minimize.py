from ast import arg
import numpy as np
from naiveb.linear import Linear
from naiveb.minimize import Minimize

class Test_Minimize:

  def test():

    func = lambda x: x[0] * np.log(x[0]) + x[0] * x[1]**2
    grad = lambda x: np.array([np.log(x[0]) + 1 + x[1]**2, 2 * x[0] * x[1]])
    hess = lambda x: np.array([[2 * x[0], - 2 * x[1]],[- 2 * x[1], 1 / x[0]]]) / (2 - 4 * x[1]**2)

    tester = Minimize(1, func, grad = grad, hess = hess, guess = np.array([1.0, 1.0]), constrain = lambda x: x[0] > 0 )
    print(tester([1,1], [2,2,100], 0.0000000001))
    print('guess: ' + str(tester.guess),
          'func: ' + str(tester.func(tester.guess)),
          'grad: ' + str(tester.grad(tester.guess)),
          'hess: ' + str(tester.hess(tester.guess)),
          sep = '\n')

    #unit testing won't allow this !!!

if (__name__ == '__main__'):
  Test_Minimize.test()
