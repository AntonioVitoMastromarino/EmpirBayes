import numpy as np

def minimize(val,gra,guess,tol=1/16,max=16):
  gradi=gra(guess)
  argument=guess-gradi
  gradient=gra(argument)
  norm=np.linalg.norm(gradi)
  adapt=[(gradi/norm,(gradi-gradient)/norm)]
  gap=tol+1
  ite=0
  while(gap>tol and ite<max):
    tem=np.array([y*(x@gradient) for (x,y) in adapt]).sum(axis=0)+gradient-np.array([x*(x@gradient) for (x,y) in adapt]).sum(axis=0)
    guess=argument-tem
    gradi=gra(guess)
    gap=val(argument)-val(guess)+np.linalg.norm(gradi)
    new=gradient-gradi
    argument=guess
    gradient=gradi
    tem-=np.array([y*(x@new) for (x,y) in adapt]).sum(axis=0)
    new-=np.array([x*(x@new) for (x,y) in adapt]).sum(axis=0)
    norm=np.linalg.norm(new)
    adapt.append(new/norm,tem/norm)
    ite+=1
  return argument
