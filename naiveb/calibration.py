import numpy as np
from naiveb.cluster import cluster

def calibration(n:int,N:int,conditional,gradient,sample,guess,prior=1,tol=1,maxiter=16,learn=1):
  #conditional(x,parameters),prior are normalized arrays of n probabilities
  #gradient(x,parameters) is an array of n parameters
  #smaple is an array of N samples
  posterior=prior
  gap=tol
  iter=0
  parameters=guess
  while(gap>tol/(n*np.sqrt(N)) and iter<maxiter):
    posterior=cluster(n,N,lambda x:conditional(x,parameters),sample,prior=posterior)
    likelihoods=np.array([(conditional(x)*posterior).sum() for x in list(sample)])
    step=np.array([np.array([gradient(sample[K],parameters)[k]*posterior[k] for k in range(n)]).sum()/likelihoods[K] for K in range(N)]).sum(axis=0)
    parameters+=learn*step
  
