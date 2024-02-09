# NaiveB

This class attempts to apply quasi-Newton method to a maximum likelihood problem where the likelihood is a weighted mixture.

>## The model: naiveb.cluster
>
>The likelihood in our problem is a mixture of a fixed number of _conditional_ likelihoods.
>
>$f\left(x,\theta,\omega\right)=\sum\_{i=1,\dots,n}\omega\_if\left(x,\theta\_i\right).$
>
>Here $n$ is the number of components of the mixture and $\{w\_i: i = 1, \dots, n\}$ are the weights. The function to maximize is the logarithm of the likelihood of a series of independent observations $\{{\rm X}\_k:k=1,\dots,{\rm N}\}$
>
>$-\phi\left(\theta,\omega\right)=\log\left(\prod\_{k=1,\dots,{\rm N}}\sum\_{i=1,\dots,n}\omega\_if\left({\rm X}\_k,\theta\_i\right)\right)=\sum\_{k=1,\dots,{\rm N}}\log\left(\sum\_{i=1,\dots,n}\omega\_if\left({\rm X}\_k,\theta\_i\right)\right).$
>
>>### Optimize the parameters 
>>
>>The first partial derivative of $\phi$ in $\partial\theta\_i$ is
>>
>>$\frac{\partial\phi}{\partial\theta\_i}=w\_i\sum\_{k=1,\dots,{\rm N}}\left(\sum\_{i=1,\dots,n}\omega\_if\left({\rm X}\_k,\theta\_i\right)\right)^{-1}\frac{\partial}{\partial\theta}f\left({\rm X}\_k,\theta\_i\right).$
>>
>>The second partial derivative in $\partial\theta\_i^2$ is
>>
>>$\frac{\partial^2\phi}{\partial\theta\_i^2}=\sum\_{k=1,\dots,{\rm N}}\left(\omega\_i\left(\sum\_{j=1,\dots,n}\omega\_jf\left({\rm X}\_k,\theta\_j\right)\right)^{-1}\frac{\partial^2}{\partial\theta^2}f\left({\rm X}\_k,\theta\_i\right)-\omega\_i^2\left(\left(\sum\_{j=1,\dots,n}\omega\_jf\left({\rm X}\_k,\theta\_j\right)\right)^{-1}\frac{\partial}{\partial\theta}f\left({\rm X}\_k,\theta\_j\right)\right)^2\right).$
>>
>>The second derivative in $\partial\theta\_i\partial\theta\_j$ is
>>
>>$\frac{\partial^2\phi}{\partial\theta\_i\partial\theta\_j}=-\omega\_i\omega\_j\sum\_{k=1,\dots,{\rm N}}\left(\sum\_{l=1,\dots,n}\omega\_lf\left({\rm X}\_k,\theta\_l\right)\right)^{-2}\left(\frac{\partial}{\partial\theta}f\left({\rm X}\_k,\theta\_i\right)\right)\left(\frac{\partial}{\partial\theta}f\left({\rm X}\_k,\theta\_j\right)\right)$
>>
>>Assume that the weights of the mixture are know. The problem is then reduced to find the minimum of a function, $\phi$, for which the first and second derivatives are available.
>
>>### Optimize the weights
>>
>>The first partial derivative in $\partial\omega\_i$ is
>>
>>$\frac{\partial\phi}{\partial\omega\_i}=\sum\_{k=1,\dots,{\rm N}}\left(\sum\_{j=1,\dots,n}\omega\_jf\left({\rm X}\_k,\theta\_j\right)\right)^{-1}f\left({\rm X\_k},\theta\_i\right)$
>>
>>so that
>>
>>$\frac{1}{\omega\_i}{\mathbb P}\left(\theta\_i,{\rm X},\omega\right)=\frac{1}{\omega\_i\rm N}\sum\_{k=1,\dots,{\rm N}}f\left(\theta\_i,{\rm X}\_k,\omega\right)=\frac{1}{\rm N}\frac{\partial\phi}{\partial\omega\_i}$
>>
>>so that if $f\left(\theta\_i,{\rm X},\omega\right)=\omega\_i$, then the gradient of $\phi$ is parallel to the constrain $\sum\_{i=1,\dots,n}\omega\_i=1$. Then we can obtain a maximum likelihood estimator recursively applying $\omega\_i^{\left(n+1\right)}=f\left(\theta\_i,{\rm X},\omega^{\left(n\right)}\right)$. The actual convergence of this method to a maximum likelihood estimator has to be, mathematically speaking, further investigated.
>
>>### Content of the module
>>
>>The module contains a class naiveb.cluster.Cluster which is initialized with:
>>- num $=n$
>>- dim $=d$ such that $\theta\in{\mathbb R}^d$
>>- func $=f:\mathcal X\times{\mathbb R}^d\rightarrow\left[0,\infty\right)$ such that $\mathrm X\_k:\Omega\rightarrow{\cal X}$
>>- grad $=\frac{\partial}{\partial\theta}f$
>>- hess $=\frac{\partial^2}{\partial\theta^2}f$
>>- prior $=\omega^{\left(0\right)}$
>>- theta $=\theta^{\left(0\right)}$
>>- gap $=\left|\omega^{\left(n+1\right)}-\omega^{\left(n\right)}\right|$ has to be set when $\omega$ is updated.
>>
>>The methods of this class are:
>>
>>- log\_like: Computes $\phi$ from a sample $\{{\rm X}\_k:k=1,\dots,{\rm N}\}$
>>
>>- grad\_log: Computes $\frac{\partial\phi}{\partial\theta}$ from a sample $\{{\rm X}\_k:k=1,\dots,{\rm N}\}$
>>
>>- inv\_hess: Computes the inverse matrix of $\frac{\partial^2\phi}{\partial\theta^2}$ from a sample $\{{\rm X}\_k:k=1,\dots,{\rm N}\}$
>>
>>- calibrator: Given a sample $\{{\rm X}\_k:k=1,\dots,{\rm N}\}$ return an object of the class naiveb.minimize.Minimize, when called this object tries to optimize $\theta$ eventually updating the weights.
>>
>>- \_\_call\_\_: Given the hyperparameters necessary to the minimization protocol described below, runs the protocol. At each iteration of the protocol $\theta$ is optimized, then, if $\left|\partial\phi\right|$ is smaller that a tollerance, the parameter $\omega$ is updated. If the gap $\left|\omega^{\left(n+1\right)}-\omega^{\left(n\right)}\right|$ is smaller than a tollerance, then condition is satisfied and the protocol returns.

>## Optimization problems: naiveb.minimize
>
>>### Random descent
>>
>>Starting from an initial guess $x^{\left(0\right)}$ and updating recursively $x^{\left(n+1\right)}\sim{\cal N}\left(x^{\left(n\right)},\varepsilon^{\left(n\right)}\right)$ as a Gaussian random variable. Then accept $x^{\left(n+1\right)}$ if the condition $\phi\left(x^{\left(n+1\right)}\right)<\phi\left(x^{\left(n\right)}\right)$ is satisfied.
>
>>### Gradient descent
>>
>>Starting from an initial guess $x^{\left(0\right)}$ and updating recursively $x^{\left(n+1\right)}=x^{\left(n\right)}-\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)$ one obtain
>>
>>$\phi\left(x^{\left(n+1\right)}\right) = \phi\left(x^{\left(n\right)}\right)-\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}-\lambda^{\left(n\right)}\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)\right)\cdot\nabla\phi\left(x^{\left(n\right)}\right)$
>>
>>for a certain series of $\lambda^{\left(n\right)}\in\left[0,1\right]$. Assume that $x^{\left(n\right)}$ is not a minimum, if the learning rate $\varepsilon^{\left(n\right)}$ is small enough the term
>>
>>$\lim\_{\varepsilon^{\left(n\right)}\rightarrow0}\nabla\phi\left(x^{\left(n\right)}-\lambda^{\left(n\right)}\varepsilon^{\left(n\right)}\nabla\phi\left(x^{\left(n\right)}\right)\right)\cdot\nabla\phi\left(x^{\left(n\right)}\right)=\left|\nabla\phi\left(x^{\left(n\right)}\right)\right|^2>0$
>>
>>is positive. That means that $\phi\left(x^{\left(n+1\right)}\right)<\phi\left(x^{\left(n\right)}\right)$, unfortunately the improvement is about $\phi\left(x^{\left(n+1\right)}\right)-\phi\left(x^{\left(n\right)}\right)\simeq\varepsilon^{\left(n\right)}\left|\nabla\phi\left(x^{\left(n\right)}\right)\right|^2$ which is small near the local minima and it is proportional to the learning rate (so a small learning rate means a slow convergence to the local minimum).
>
>>### Newton's method
>>
>>The Newton's method does not actually look for the local minimum of $\phi$, but for a zero of $\psi=\nabla\phi$. Say that $\psi\left(x\_0\right)=0$ and that the initial guess $x^{\left(n\right)}$ is near enough to $x\_0$, then
>>
>>$0=\psi\left(x\_0\right)=\psi\left(x^{\left(n\right)}\right)+\nabla\psi\left(x^{\left(n\right)}\right)\left(x\_0 - x^{\left(n\right)}\right)+o\left|x\_0-x^{\left(n\right)}\right|$
>>
>>so that
>>
>>$x\_0=x^{\left(n\right)}-\nabla\psi\left(x^{\left(n\right)}\right)^{-1}\left(\psi\left(x^{\left(n\right)}\right)+o\left|x\_0-x^{\left(n\right)}\right|\right).$
>>
>>Defining recursively
>>
>>$x^{\left(n+1\right)}=x^{\left(n\right)}-\nabla\psi\left(x^{\left(n\right)}\right)^{-1}\left(\psi\left(x^{\left(n\right)}\right)\right)$
>>
>>one obtain that
>>
>>$\left|x^{\left(n+1\right)}-x\_0\right|=\nabla\psi\left(x^{\left(n\right)}\right)^{-1}o\left|x\_0-x^{\left(n\right)}\right|$
>>
>>so that if $\nabla\psi$ (*i.e.: the Hessian of $\phi$*) is non-singular in $x\_0$, then the convergence of $x^{\left(n\right)}$ to $x\_0$ is super-linear, if moreover $\psi$ admits second derivative in $x\_0$ (*i.e.: the Hessian of $\phi$ is differentiable in $x\_0$*), then
>>
>>$\log\left(-\log\left|x^{\left(n\right)}-x\_0\right|\right)\simeq\log\left(2\right)n.$
>
>>### Our protocol for the minimization
>>
>>It is desirable to use the Newton's method to find $\arg\min\phi$, but to ensure that series $x^{\left(n\right)}$ converges to a local minimum and not to a local maximum, or a saddle point, we alternate the two methods. Moreover at each iteration we check that $\phi\left(x^{\left(n+1\right)}\right)<\phi\left(x^{\left(n\right)}\right)$ and accept $x^{\left(n+1\right)}$ only if the condition is satisfied. The problem of chosing $\varepsilon^{\left(n\right)}$ and of deciding how to alternate the two methods is challenging.
>>
>>Our arbitrary protocol consists in repeating what follows until a stopping condition is satisfied:
>>- three integers $n\_{rd},n\_{gd},n\_{nt}$ and two numbers $\varepsilon\_{rd},\varepsilon\_{rd}$ are given
>>- for $n=0,\dots,n\_{rd}-1$:
>>- - update the guess $x^{\left(n\right)}$ with random descent, $\varepsilon^{\left(n\right)}=\varepsilon\_{rd}$
>>- - if the proposal is refused, just set $x^{\left(n+1\right)}=x^{\left(n\right)}$ 
>>- - otherwise, slightly increase $\varepsilon\_{rd}$ 
>>- - if a stopping condition is satisfied return
>>- set $x^{\left(0\right)}=x^{\left(n\_{rd}\right)}$
>>- for $n=0,\dots,n\_{gd}-1$:
>>- - update the guess $x^{\left(n\right)}$ with gradiendt descent, $\varepsilon^{\left(n\right)}=\varepsilon\_{gd}$
>>- - if the proposal is refused, just set $x^{\left(n+1\right)}=x^{\left(n\right)}$ and reduce $\varepsilon\_{gd}$
>>- - otherwise slightly increase $\varepsilon\_{gd}$
>>- - if a stopping condition is satisfied return
>>- set $x^{\left(0\right)}=x^{\left(n\_{gd}\right)}$
>>- for $n=0,\dots,n\_{nt}-1$:
>>- - update the guess $x^{\left(n\right)}$ with gradiendt descent, $\varepsilon^{\left(n\right)}=\varepsilon\_{gd}$
>>- - if the proposal is refused:
>>- - - set $x^{\left(n\_{nt}\right)}=x^{\left(n\right)}$
>>- - - slightly increase $n\_{gd}$
>>- - - slightly decrease $n\_{rd}$
>>- - otherwise, if $n=n\_{ns}$:
>>- - - decrease $\varepsilon\_{rd}$
>>- - - increase $n\_{rd}$
>>- - - decrease $n\_{gd}$ 
>>- - if a stopping condition is satisfied return
>>- set $x^{\left(0\right)}=x^{\left(n\_{nt}\right)}$
>>
>>In this way:
>>- $\varepsilon\_{gd}$ is always near enough to its most effective value.
>>- $\varepsilon\_{rd}$ is almost always large, so that the random descend can search for more advantageous areas in case getting stuck to a local minimum.
>>- $\varepsilon\_{rd}$ suddenly decays when the guess gets closer to a zero of the gradient, to prevent converging to a possible saddle point.
>>- $n\_{rd}{\lt}n\_{gd}$ until a neighborhood of a zero of the gradient is reached.
>>- $n\_{gd}{\lt}n\_{rd}$ when close to a zero of the gradient, so that the gradient descend is replaced by the more effective newton method and the random descend prevents convergence to saddle points.
>>- The increase of $n\_{rd}$ near to saddles allows $\varepsilon\_{rd}$ to increase again during the iterations, balancing its decay.
>
>>### Content of the module
>>
>>The module contains a class naiveb.minimize.Minimize which is initialized with:
>>- dim $=d$ such that $\phi:\mathbb R^d\rightarrow\mathbb R$
>>- guess $=x^{\left(0\right)}$
>>- func $=\phi\left(x\right)$
>>- grad $=\nabla\phi\left(x\right)$
>>- hess $=\nabla^2\phi\left(x\right)^{-1}$
>>- constrain is equivalent to $\phi\left(x\right)<+\infty$
>>- update is called at the end of the method \_\_call\_\_
>>- grad\_avail: is True when grad is initialized. If False, grad is inferred with the class Linear (in development)
>>- hess\_avail: is True when hess is initialized. If False, hess is inferred with the class Linear (in development)
>>
>>The methods of this class are:
>>
>>- compute: Takes no input. Just compute grad in guess. If grad is not initialized, then it inferres the gradient locally (in development)
>>
>>- attempt: Given a step $x^{\left(n\right)}-x^{\left(n+1\right)}$ checks if the proposal can be accepted, in case replace guess. If grad is not initialized, then it updates grad (in development). If hess is not initialized, then it updates hess (in development)
>>
>>- nt\_step: Computes $x^{\left(n\right)}-x^{\left(n+1\right)}$ for the Newton method and calls attempt. If the proposal is refused tries to use the same step with opposite sign.
>>
>>- gd\_step: Given $\varepsilon\_{gd},\varepsilon\_{rd}$ computes $x^{\left(n\right)}-x^{\left(n+1\right)}$ for the gradient descent and for the random descent, then calls attempt. If the proposal is refused tries to use the same step with opposite sign. The use of this method with both parameters different from zero has not been tested.
>>
>>- \_\_call\_\_: Given $n\_{rd},n\_{gd},n\_{nt}, \varepsilon\_{rd},\varepsilon\_{rd}$ and a tollerance, repeat one cycle of the protocol described above, returns the number of accepted proposals and the number of the refused proposals so that the user can tune them at their taste (other hyperparameters will be available for further flexibility).
>>
>>- protocol: Given $n\_{rd},n\_{gd},n\_{nt}, \varepsilon\_{rd},\varepsilon\_{rd}$ and a tollerance, repeatedly calls the method \_\_call\_\_ adjusting them at each iteration. It never stops until the stopping condition is satisfied and has no maximum number of iterations, so a limit has to be set as optional argument in the condition.
>
>## Inferring the gradient and the Hessian: naiveb.linear
>
>If only the gradient of $\phi$ is available, then the Hessian can be inferred as
>
>$\nabla^2\phi\left(y^{\left(n+1\right)}\right)^{-1}\left(\nabla\phi\left(x^{\left(n+1\right)}\right)-\nabla\phi\left(x^{\left(n\right)}\right)\right)=x^{\left(n+1\right)}-x^{\left(n\right)}$
>
>If the gradient is not available, it can be inferred by
>
>$\nabla\phi\left(y^{\left(n+1\right)}\right)\left(x^{\left(n+1\right)}-x^{\left(n\right)}\right)=\phi\left(x^{\left(n+1\right)}\right)-\phi\left(x^{\left(n\right)}\right).$
>
>While the Hessian (this is CHALLENGING: in development)
>
>>### Content of the module
>>
>>The module contains a class naiveb.linear.Linear which is initialized with:
>>
>>- (in development)
>>
>>The methods of this class are:
>>
>>- (in development)
