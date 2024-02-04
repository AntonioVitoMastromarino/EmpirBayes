from ast import arg
import numpy as np
from naiveb.linear import Linear
from naiveb.minimize import Minimize
from sys import argv

class Test_Minimize:

  def test(x_toll, y_toll, max_iter):

    func = lambda x: x * np.log(x)
    grad = lambda x: np.log(x) + 1
    hess = lambda x: x

    tester = Minimize(1, func, grad = grad, hess = hess, guess = np.array([1.0]) )
    tester(x_toll, y_toll, max_iter)
    print('x_gap: ' + str(tester.x_gap),
          'y_gap: ' + str(tester.y_gap),
          'guess: ' + str(tester.guess),
          'func: ' + str(tester.func(tester.guess)),
          'grad: ' + str(tester.grad(tester.guess)),
          'hess: ' + str(tester.hess(tester.guess)),
          'iter: ' + str(tester.iter))

    #unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, x_toll, y_toll, max_iter = argv
  Test_Minimize.test(np.float64(x_toll), np.float64(y_toll), int(max_iter))
