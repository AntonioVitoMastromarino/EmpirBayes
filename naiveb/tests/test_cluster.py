import numpy as np
from naiveb.cluster import Cluster
from sys import argv

class Test_Cluster:

  def test(num, epsilon, samples):

    theta = np.random.rand([num,2])

    def like(x, theta):
      return np.exp(- (x - theta[0])**2 / (2 * theta[1]))
    
    def part(x, theta):
      st = (x - theta[0]) * like(x, theta) / theta[1]
      nd = (x - theta[0])**2 * like(x, theta) / (2 * theta[1]**2)
      return np.array([st, nd])

    def hess(x, thets):
      stst = ((x - theta[0]) * part(x, theta)[0] - like(x, theta)) / theta[1]
      stnd = (x - theta[0]) * (part(x, theta)[1] - like(x, theta) / theta[1]) / theta[1]
      ndst = (x - theta[0]) * ((x - theta[0]) * part(x, theta)[0] / 2 - like(x, theta)) / theta[1]**2
      ndnd = (x - theta[0])**2 * part(x, theta)[1] / (2 * theta[1]**2) - (x - theta[0])**2 * like(x, theta) / theta[1]**3
      assert stnd == ndst
      return np.array([[stst, stnd], [ndst, ndnd]])

    clu = Cluster(num, 1, like, part, hess, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), np.random.rand(num))
    X = epsilon * np.random.randn(samples) + np.random.choice(theta, p = [0.5, 0.25, 0.25])
    tester = clu.calibrator(X)
    tester(epsilon, epsilon, samples)
    print('guess: ' + str(tester.guess),
          'theta: ' + str(theta),
          'posterior: ' + str(clu.prior))

    #unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, num, epsilon, samples = argv
  Test_Cluster.test(int(num), np.float64(epsilon), int(samples))
