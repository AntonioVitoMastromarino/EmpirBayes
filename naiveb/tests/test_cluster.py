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
      nd = like(x, theta) * (theta[1] * (x - theta[0])**2 / (2 * np.exp(theta[1])) - 1)
    # st = (x - theta[0]) * like(x, theta) / theta[1]
    # nd = ((x - theta[0])**2 - 2 * theta[1]) * like(x, theta) / (2 * theta[1]**2)
      return np.array([st, nd])

    def hess(x, theta):
      st = (part(x, theta)[0] * (x - theta[0]) - like(x, theta)) / np.exp(theta[1])
      nd = part(x, theta)[1] * (x - theta[0]) / np.exp(theta[1]) - theta[1] * part(x, theta)[0]
      rd = part(x, theta)[1] * (theta[1] * (x - theta[0])**2 / (2 * np.exp(theta[1])) - 1) + like(x, theta) * (x - theta[0])**2 * (1 - theta[1]**2) / (2 * np.exp(theta[1]))
    # st = ((x - theta[0]) * part(x, theta)[0] - like(x, theta)) / theta[1]
    # nd = (x - theta[0]) * ((x - theta[0])**2 / (2 * theta[1]) - 2) * like(x, theta) / theta[1]**2
    # rd = ((((x - theta[0])**2 - 2 * theta[1])**2 + 4 * theta[1]**2 - 4 * (x - theta[0])**2 * theta[1]) - 4 * theta[1]**3) * like(x, theta) / (4 * theta[1]**4)
      return np.array([[st, nd], [nd, rd]])

    # There must be something wrong in these functions
    # try substitute theta[1] with np.exp(theta[1])


    p = np.exp(- np.array(range(1,num + 1)))
    p /= p.sum()
    clu = Cluster(num, 2, like, part, hess, p, guess)
    sample_theta = [list(THETA)[np.random.choice(range(num), p = p)] for k in range(samples)]
    X = [t[1] * np.random.randn() + t[0] for t in sample_theta]
    tester = clu.calibrator(X)
    res = True

    print('theta:', str(THETA),
          'posterior:', str(clu.prior),
          sep = '\n', flush = True)

    print('guess:', str(tester.guess),
          'cluster:', str(clu.theta), 
          sep = '\n', flush = True)

    while res:
      old_guess = tester.guess
      tester([0, 0.001], [0, 100, 100], 0.01, 100)
      if np.linalg.norm(tester.grad(tester.guess)) < 0.1:
        res = False
        print('found')
      elif (tester.guess == old_guess).all():
        res = False

    print('guess:', str(tester.guess),
          'cluster:', str(clu.theta), 
          sep = '\n', flush = True)

    tester.nt_step()

    print('guess:', str(tester.guess),
          'cluster:', str(clu.theta), 
          sep = '\n', flush = True)
    
    # unit testing won't allow this !!!

if (__name__ == '__main__'):
  script_name, num, samples = argv
  Test_Cluster.test(int(num), int(samples))
