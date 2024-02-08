# NaiveB
This class attempts to apply quasi-Newton methods to a maximum likelihood problem. The likelihood in our problem is a mixture of a fixed number of 'conditional' likelihoods.

$f\left(x,\theta_i:i=1,\dots,n\right)=\sum_{i=1,\dots,n}\omega_if\left(x,\theta_i\right).$

Here $n$ is the number of components of the mixture and $\{w_i: i = 1, \dots, n\}$ are the weights. The function to maximize is the logarithm of the likelihood of a series of independent observations $\{{\rm X}_k:k=1,\dots,{\rm N}\}$

$\log\left(\prod_{k=1,\dots,{\rm N}}\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right)=\sum_{k=1,\dots,{\rm N}}\log\left(\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right).$

The first partial derivative of this in $\partial\theta_i$ is

$w_i\sum_{k=1,\dots,{\rm N}}\left(\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right)^{-1}\frac{\partial}{\partial\theta}f\left({\rm X}_k,\theta_i\right)$

the second partial derivative in $\partial\theta_i^2$ is

$w_i\sum_{k=1,\dots,{\rm N}}\left(\left(\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right)^{-1}\frac{\partial^2}{\partial\theta^2}f\left({\rm X}_k,\theta_i\right)-\left(\left(\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right)^{-1}\frac{\partial}{\partial\theta}f\left({\rm X}_k,\theta_i\right)\right)^2\right)$

while the second derivative in $\partial\theta_i\partial\theta_j$ is

$-w_i\sum_{k=1,\dots,{\rm N}}\left(\sum_{i=1,\dots,n}f\left({\rm X}_k,\theta_i\right)\right)^{-2}\left(\frac{\partial}{\partial\theta}f\left({\rm X}_k,\theta_i\right)\right)\left(\frac{\partial}{\partial\theta}f\left({\rm X}_k,\theta_j\right)\right)$

Assume that the weights of the mixture are know. The problem is then reduced to find the maximum of a function for which the first and second derivatives are available.

>## Optimization problems
>
>>### Gradient descent
>>
>>Starting from an initial guess $x^{\left(0\right)}$ and updating recursively $x^{\left(n+1\right)}=x^{\left(n\right)}-\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)$ one obtain
>>
>>$\phi\left(x^{\left(n+1\right)}\right) = \phi\left(x^{\left(n\right)}\right)-\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}-\lambda^{\left(n\right)}\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)\right)\cdot\nabla\phi\left(x^{\left(n\right)}\right)$
>>
>>for a certain series of $\lambda^{\left(n\right)}\in\left[0,1\right]$. Assume that $x^{\left(n\right)}$ is not a minimum, if the learning rate $\varepsilon^{\left(n\right)}$ is small enough the term
>>
>>$\lim_{\varepsilon^{\left(n\right)}\rightarrow0}\nabla\phi\left(x^{\left(n\right)}-\lambda^{\left(n\right)}\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)\right)\cdot\nabla\phi\left(x^{\left(n\right)}\right)=\left|\nabla\phi\left(x^{\left(n\right)}\right)\right|^2>0$
>>
>>is positive. That means that $\phi\left(x^{\left(n+1\right)}\right)<\phi\left(x^{\left(n\right)}\right)$, unfortunately the improvement is about $\phi\left(x^{\left(n+1\right)}\right)-\phi\left(x^{\left(n\right)}\right)\simeq\varepsilon^{\left(n\right)}\left|\nabla\phi\left(x^{\left(n\right)}\right)\right|^2$ which is small near the local minima and it is proportional to the learning rate (so a small learning rate means a slow convergence to the local minimum).
>
>>### Newton's method
>>
>>The Newton's method does not actually looks for the local minimum of $\phi$, but for a zero of $\psi=\nabla\phi$. Say that $\psi\left(x_0\right)=0$ and that the initial guess $x^{\left(n\right)}$ is near enough to $x_0$, then
>>
>>$0=\psi\left(x_0\right)=\psi\left(x^{\left(n\right)}\right)+\nabla\psi\left(x^{\left(n\right)}\right)\left(x_0 - x^{\left(n\right)}\right)+o\left|x_0-x^{\left(n\right)}\right|$
>>
>>so that
>>
>>$x_0=x^{\left(n\right)}-\nabla\psi\left(x^{\left(n\right)}\right)^{-1}\left(\psi\left(x^{\left(n\right)}\right)+o\left|x_0-x^{\left(n\right)}\right|\right).$
>>
>>Defining recursively
>>
>>$x^{\left(n+1\right)}=x^{\left(n\right)}-\nabla\psi\left(x^{\left(n\right)}\right)^{-1}\left(\psi\left(x^{\left(n\right)}\right)\right)$
>>
>>one obtain that
>>
>>$\left|x^{\left(n+1\right)}-x_0\right|=\nabla\psi\left(x^{\left(n\right)}\right)^{-1}o\left|x_0-x^{\left(n\right)}\right|$
>>
>>so that if $\nabla\psi$ (*i.e.: the Hessian of $\phi$*) is non-singular in $x_0$, then the convergence of $x^{\left(n\right)}$ to $x_0$ is super-linear, if moreover $\psi$ admits second derivative in $x_0$ (_i.e.: the Hessian of $\phi$ is differentiable in $x_0$_), then
>>
>>$\log\left(-\log\left|x^{\left(n\right)}-x_0\right|\right)\simeq\log\left(2\right)n.$
>
>It is desirable to use the Newton's method to find $\argmin\phi$, but to ensure that series $x^{\left(n\right)}$ converges to a local minimum and not to a local maximum, or a saddle point, we alternate the two methods. Moreover at each iteration we check that $\phi\left(x^{\left(n+1\right)}\right)<\phi\left(x^{\left(n\right)}\right)$ and accept $x^{\left(n+1\right)}$ only if the condition is satisfied. The problem of chosing $\varepsilon^{\left(n\right)}$ and of deciding how to alternate the two methods is challenging.
