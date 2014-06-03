MGP
===

This is a `MATLAB` implementation of the "marginal GP" (MGP) described
in:
> Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
> Embeddings for Gaussian Processes. (2014). 30th Conference on
> Uncertainty in Artificial Intellignece (UAI 2014).

Suppose we have a Gaussian process model on a latent function ![f][1]:

![p(f | \theta) = GP(f; \mu(x; \theta), K(x, x'; \theta))][2]

where ![\theta][3] are the hyperparameters of the model. Suppose we
have a dataset ![D = (X, y)][4] of observations and a test point
![x*][5]. This function returns the mean and variance of the
approximate marginal predictive distributions for the associated
observation value ![y*][6] and latent function value ![f*][7]:

![p(y* | x*, D) = \int p(y* | x*, D, \theta) p(\theta | D) d\theta][8]

![p(f* | x*, D) = \int p(f* | x*, D, \theta) p(\theta | D) d\theta][9]

where we have marginalized over the hyperparameters ![\theta][3].

Notes
-----

This code is only appropriate for GP regression! Exact inference
with a Gaussian observation likelihood is assumed.

The MGP approximation requires that the provided hyperparameters be
the MLE hyperparameters:

![\hat{\theta} = argmax_\theta log p(y | X, \theta)][10]

or, if using a hyperparameter prior ![p(\theta)][11], the MAP
hyperparameters:

![\hat{\theta} = argmax_\theta log p(y | X, \theta) + log p(\theta)][12]

This function does not perform the maximization over ![\theta][3] but
rather assumes that the given hyperparameters represent
![\hat{\theta}][13].

Dependencies
------------

This code is written to be interoperable with the GPML MATLAB
toolbox, available here:

http://www.gaussianprocess.org/gpml/code/matlab/doc/

The GPML toolbox must be in your MATLAB path for this function to
work. This function also depends on the `gpml_extensions` repository,
available here:

https://github.com/rmgarnett/gpml_extensions/

which must also be in your MATLAB path.

Usage
-----

The usage of `mgp.m` is identical to the `gp.m` function from the GPML
toolkit in prediction mode. See `mgp.m` for more information.

A demo is provided in `demo/demo.m`.

[1]: http://latex.codecogs.com/svg.latex?f
[2]: http://latex.codecogs.com/svg.latex?p(f%20%5Cmid%20%5Ctheta)%20%3D%20%5Cmathcal%7BGP%7D%5Cbigl(f%3B%20%5Cmu(x%3B%20%5Ctheta)%2C%20K(x%2C%20x%27%3B%20%5Ctheta)%5Cbigr)
[3]: http://latex.codecogs.com/svg.latex?%5Ctheta
[4]: http://latex.codecogs.com/svg.latex?%5Cmathcal%7BD%7D%20%3D%20(X%2C%20y)
[5]: http://latex.codecogs.com/svg.latex?x%5E%5Cast
[6]: http://latex.codecogs.com/svg.latex?y%5E%5Cast
[7]: http://latex.codecogs.com/svg.latex?f%5E%5Cast
[8]: http://latex.codecogs.com/svg.latex?p(y%5E%5Cast%20%5Cmid%20x%5E%5Cast%2C%20%5Cmathcal%7BD%7D)%20%3D%20%5Cint%20p(y%5E%5Cast%20%5Cmid%20x%5E%5Cast%2C%20%5Cmathcal%7BD%7D%2C%20%5Ctheta)%20p(%5Ctheta%20%5Cmid%20%5Cmathcal%7BD%7D)%20%5C%2C%20%5Cmathrm%7Bd%7D%5Ctheta
[9]: http://latex.codecogs.com/svg.latex?p(f%5E%5Cast%20%5Cmid%20x%5E%5Cast%2C%20%5Cmathcal%7BD%7D)%20%3D%20%5Cint%20p(f%5E%5Cast%20%5Cmid%20x%5E%5Cast%2C%20%5Cmathcal%7BD%7D%2C%20%5Ctheta)%20p(%5Ctheta%20%5Cmid%20%5Cmathcal%7BD%7D)%20%5C%2C%20%5Cmathrm%7Bd%7D%5Ctheta
[10]: http://latex.codecogs.com/svg.latex?%5Chat%7B%5Ctheta%7D%20%3D%20%5Coperatorname*%7Barg%5C%2Cmax%7D_%7B%5Ctheta%7D%20%5Clog%20p(y%20%5Cmid%20X%2C%20%5Ctheta)
[12]: http://latex.codecogs.com/svg.latex?p(%5Ctheta)
[12]: http://latex.codecogs.com/svg.latex?%5Chat%7B%5Ctheta%7D%20%3D%20%5Coperatorname*%7Barg%5C%2Cmax%7D_%7B%5Ctheta%7D%20%5Clog%20p(y%20%5Cmid%20X%2C%20%5Ctheta)%20%2B%20%5Clog%20p(%5Ctheta)
[13]: http://latex.codecogs.com/svg.latex?%5Chat%7B%5Ctheta%7D
