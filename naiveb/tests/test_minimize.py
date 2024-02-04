from ast import arg
import numpy as np
from naiveb.linear import Linear
from naiveb.minimize import Minimize
from sys import argv

class Test_Minimize:

  def test(max_toll, max_iter):

    func = lambda x,y: x * np.log(x)
    grad = lambda x: np.log(x) + 1
    hess = lambda x: 1 / x

    tester = Minimize(1, func, grad = grad, hess = hess, guess = 1)
    tester(max_toll, max_iter)
    print(tester.gap, tester.guess, tester.iter)
    #unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, max_toll, max_iter = argv
  print(script_name)
  Test_Minimize.test(float(max_toll), int(max_iter))