import numpy as np
from naiveb.cluster import Cluster
from sys import argv

class Test_Cluster:

  def test(num, samples):

    theta = np.block([[np.random.rand(num, 1), 0.0001 * np.ones([num, 1])]])
    guess = np.block([[np.random.rand(num, 1), np.ones([num, 1])]])

    def like(x, theta):
      return np.exp(- (x - theta[0])**2 / (2 * theta[1]))
    
    def part(x, theta):
      st = (x - theta[0]) * like(x, theta) / theta[1]
      nd = (x - theta[0])**2 * like(x, theta) / (2 * theta[1]**2)
      return np.array([st, nd])

    def hess(x, theta):
      stst = ((x - theta[0]) * part(x, theta)[0] - like(x, theta)) / theta[1]
      stnd = (x - theta[0]) * (part(x, theta)[1] - like(x, theta) / theta[1]) / theta[1]
      ndst = (x - theta[0]) * ((x - theta[0]) * part(x, theta)[0] / 2 - like(x, theta)) / theta[1]**2
      ndnd = (x - theta[0])**2 * part(x, theta)[1] / (2 * theta[1]**2) - (x - theta[0])**2 * like(x, theta) / theta[1]**3
      assert stnd == ndst
      return np.array([[stst, stnd], [ndst, ndnd]])

    clu = Cluster(num, 2, like, part, hess, np.ones(num)/num, guess)
    p = np.exp(- np.array(range(1,num + 1)))
    p /= p.sum()
    sample_theta = [list(theta)[np.random.choice(range(num), p = p)] for k in range(samples)]
    X = [t[1] * np.random.randn() + t[0] for t in sample_theta]
    tester = clu.calibrator(X)
    res = True
    while res:
      print('guess:', str(tester.guess),
#           'cluster:', str(clu.theta), 
#           'theta:', str(theta),
            'posterior:', str(clu.prior),
            sep = '\n', flush = True)
      tester([0.00001, 0.00001], [100, 100, 10], 0.1, 10)
      if np.linalg.norm(tester.grad(tester.guess)) < 0.1:
        res = False

    # unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, num, samples = argv
  Test_Cluster.test(int(num), int(samples))
