import numpy as np

def normalize(x): return x/x.sum()

def cluster(n:int,N:int,conditional,sample,prior=1,tol=1,maxiter=16):
  #conditional(x),prior are normalized arrays of n probabilities
  #smaple is an array of N samples
  posterior=prior
  classification=np.ndarray([N,n])
  gap=tol
  iter=0
  while(gap>tol/(n*np.sqrt(N)) and iter<maxiter):
    likelihoods=np.array([(conditional(x)*posterior).sum() for x in list(sample)])
    classification=np.array([conditional(sample[K])*posterior/likelihoods[K] for K in range(N)])
    gap=((posterior-classification.sum(axis=0))**2).sum()
    posterior=classification.sum(axis=0)
    iter+=1
  return posterior
