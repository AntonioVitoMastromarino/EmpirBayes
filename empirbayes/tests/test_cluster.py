import numpy as np
from naiveb.cluster import Cluster
from sys import argv

class Test_Cluster:

  def test(num, samples):

    THETA = np.block([[np.random.rand(num, 1), - 3 * np.ones([num, 1])]])
    guess = np.block([[np.random.rand(num, 1), + 3 * np.ones([num, 1])]])

    def like(x, theta):
      return np.exp(- (x - theta[0])**2 / (2 * np.exp(theta[1])) - theta[1]) / (2 * np.pi)
    
    def part(x, theta):
      st = like(x, theta) * (x - theta[0]) / np.exp(theta[1])
      nd = like(x, theta) * ((x - theta[0])**2 / (2 * np.exp(theta[1])) - 1)
    # st = (x - theta[0]) * like(x, theta) / theta[1]
    # nd = ((x - theta[0])**2 - 2 * theta[1]) * like(x, theta) / (2 * theta[1]**2)
      return np.array([st, nd])

    def hess(x, theta):
      st = like(x, theta) * ((x - theta[0])**2 / np.exp(theta[1]) - 1) / np.exp(theta[1])
      nd = like(x, theta) * ((x - theta[0])**2 / (2 * np.exp(theta[1])) - 2) * (x - theta[0]) / np.exp(theta[1])
      rd = like(x, theta) * ((x - theta[0])**2 / (2 * np.exp(theta[1])) - 1) * ((x - theta[0])**2 / (2 * np.exp(theta[1])) - 2)
    # st = ((x - theta[0]) * part(x, theta)[0] - like(x, theta)) / theta[1]
    # nd = (x - theta[0]) * ((x - theta[0])**2 / (2 * theta[1]) - 2) * like(x, theta) / theta[1]**2
    # rd = ((((x - theta[0])**2 - 2 * theta[1])**2 + 4 * theta[1]**2 - 4 * (x - theta[0])**2 * theta[1]) - 4 * theta[1]**3) * like(x, theta) / (4 * theta[1]**4)
      return np.array([[st, nd], [nd, rd]])


    p = np.exp(- np.array(range(1,num + 1)))
    p /= p.sum()
    clu = Cluster(num, 2, like, part, hess, p, guess)
    sample_theta = [list(THETA)[np.random.choice(range(num), p = p)] for k in range(samples)]
    X = [t[1] * np.random.randn() + t[0] for t in sample_theta]
    tester = clu.calibrator(X)
    res = True
    toll = 0.01
    rates = [0.1, 0.1]
    steps = [500, 500, 10]

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
    
    while res:
      old_prior = clu.prior
      succ, fail = tester(rates, steps, toll**2)
      print(succ, fail)

      if (succ[0] > 0):
        rates[0] *= 1 + succ[0] / (steps[0] + 1)

      if (succ[1] > 0):
        rates[1] *= (2 * succ[1] + 1) / (steps[1] + 1)

      if (fail[2] > 0):
        steps[0], steps[1] = rot(steps[0], steps[1], C)
      else:
        steps[1], steps[0] = rot(steps[1], steps[0], C)
        rates[0] /= 2**steps[0]
        if (np.linalg.norm(clu.prior - old_prior) < toll) and (np.linalg.norm(tester.grad(tester.guess)) < toll) and (succ[2] < steps[2]):
          res = False

    print('theta:', str(THETA.transpose()),
          'weights:', str(p),
          sep = '\n', flush = True)
    
    print('cluster:', str(clu.theta.transpose()),
          'posterior:', str(clu.prior),
          sep = '\n', flush = True)
    
    # unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, num, samples = argv
  Test_Cluster.test(int(num), int(samples))
