% MGP approximates marginal GP predictions.
%
% Suppose we have a Gaussian process model on a latent function f:
%
%   p(f | \theta) = GP(f; \mu(x; \theta), K(x, x'; \theta)),
%
% where \theta are the hyperparameters of the model. Suppose we have a
% dataset D = (x, y) of observations and a set of test points x*. This
% function returns the mean and variance of the approximate marginal
% predictive distributions for the observations y* and latent function
% values f*:
%
%   p(y* | x*, D) = \int p(y* | x*, D, \theta) p(\theta | D) d\theta
%   p(f* | x*, D) = \int p(f* | x*, D, \theta) p(\theta | D) d\theta
%
% marginalizing over hyperparameters \theta. The approximate posterior
% is derived using he "MGP" approximation described in
%
%   Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
%   Embeddings for Gaussian Processes. (2013). arXiv:1310.6740 [stat.ML]
%
% Notes
% -----
%
% This code is only appropriate for GP regression! Exact inference
% with a Gaussian likelihood is assumed.
%
% The MGP approximation requires that the provided hyperparameters
% be the MAP hyperparameters:
%
%   \hat{theta} = argmax_\theta p(y | x, \theta).
%
% This function does not perform the maximization but rather
% assumes that the given hyperparameters represent \hat{theta}.
%
% Dependencies
% ------------
%
% This code is written to be interoperable with the GPML MATLAB
% toolbox, available here:
%
%   http://www.gaussianprocess.org/gpml/code/matlab/doc/
%
% The GPML toolbox must be in your MATLAB path for this function to
% work. This function also depends on the gpml_extensions repository,
% available here:
%
%   https://github.com/rmgarnett/gpml_extensions/
%
% Usage
% -----
%
% The usage of mgp.m is identical to the gp(...) function from the
% GPML toolkit in test mode:
%
%   [y_star_mean, y_star_variance, f_star_mean, f_star_variance, ...
%    log_probabilities, posterior] = ...
%        mgp(hyperparameters, inference_method, mean_function, ...
%            covariance_function, likelihood, x, y, x_star, y_star)
%
% Inputs
% ------
%
%       hyperparameters: a GPML hyperparameter struct containing the
%                        MAP hyperparameters
%      inference_method: a GPML inference method (note: infExact is assumed!)
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood (note: likGauss is assumed!)
%                     x: training observation locations (n x D)
%                x_star:     test observation locations (n_star x D)
%                     y: training observation values (n x 1)
%                y_star:     test observation values (n_star x 1) (optional)
%
% Outputs
% -------
%
%         y_star_mean: the approximate   E[y* | x*, D]
%     y_star_variance: the approximate Var[y* | x*, D]
%         f_star_mean: the approximate   E[f* | x*, D]
%     f_star_variance: the approximate Var[f* | x*, D]
%   log_probabilities: if test observation values y_star are
%                      provided, a vector containing the
%                      approximate log predictive probabilities
%
%                        log p(y* | x*, D)
%
%           posterior: a GPML posterior struct corresponding to the
%                      provided training data and MAP hyperparameters
%
% See also GP.
%
% Copyright (c) 2014 Roman Garnett.

function [y_star_mean, y_star_variance, f_star_mean, f_star_variance, ...
          log_probabilities, posterior] = mgp(hyperparameters, ~, ...
          mean_function, covariance_function, ~, x, y, x_star, y_star)

  num_test = size(x_star, 1);

  % convenience handles
  mu = @(varargin) feval(mean_function{:},       hyperparameters.mean, varargin{:});
  K  = @(varargin) feval(covariance_function{:}, hyperparameters.cov,  varargin{:});

  % find GP posterior and Hessian of log likelihood evaluated at the MAP
  % \theta
  [posterior, ~, ~, HnlZ] = exact_inference(hyperparameters, ...
          mean_function, covariance_function, [], x, y);

  % find the predictive distribution conditioned on the MAP \theta
  [~, ~, f_star_mean, f_star_variance] = ...
      gp(hyperparameters, [], mean_function, covariance_function, ...
         [], x, posterior, x_star);

  % precompute k*' [ K + \sigma^2 I ]^{-1}; it's used a lot
  k_star = K(x, x_star);

  noise_variance = exp(2 * hyperparameters.lik);

  % handle different posterior parameterizations
  if (is_chol(posterior.L))
    % high-noise parameterization: posterior.L contains chol(K / sigma^2 + I)

    k_star_V_inv = solve_chol(posterior.L, k_star)' / noise_variance;
  else
    % low-noise parameterization: posterior.L contains -inv(K + \sigma^2 I)

    k_star_V_inv = -k_star' * posterior.L;
  end

  % for each hyperparameter \theta_i, we must compute:
  %
  %   d mu_{f | D}(x*; \theta) / d \theta_i and
  %   d  V_{f | D}(x*; \theta) / d \theta_i

  num_hyperparameters = size(HnlZ.H, 1);

  df_star_mean     = zeros(num_test, num_hyperparameters);
  df_star_variance = zeros(num_test, num_hyperparameters);

  % partials with respect to covariance hyperparameters
  for i = 1:numel(HnlZ.covariance_ind)
    dK           = K(x,      [],     i);
    dk_star      = K(x,      x_star, i);
    dk_star_star = K(x_star, 'diag', i);

    df_star_mean(:, HnlZ.covariance_ind(i)) = ...
        (dk_star' - k_star_V_inv * dK) * posterior.alpha;

    df_star_variance(:, HnlZ.covariance_ind(i)) = ...
        dk_star_star - product_diag(k_star_V_inv, 2 * dk_star - dK * k_star_V_inv');
  end

  % partial with respect to likelihood hyperparameter log(\sigma)
  df_star_mean(:, HnlZ.likelihood_ind) = ...
      -2 * noise_variance * k_star_V_inv * posterior.alpha;

  df_star_variance(:, HnlZ.likelihood_ind) = ...
      2 * noise_variance * product_diag(k_star_V_inv, k_star_V_inv');

  % partials with respect to mean hyperparameters
  for i = 1:numel(HnlZ.mean_ind)
    dm      = mu(x,      i);
    dm_star = mu(x_star, i);

    df_star_mean(:, HnlZ.mean_ind(i)) = dm_star - k_star_V_inv * dm;
  end

  % we inflate the predictive variance to match MGP approximation
  f_star_variance = ...
      (4 / 3) * f_star_variance + ...
      product_diag(df_star_mean,     HnlZ.H \ df_star_mean') + ...
      product_diag(df_star_variance, HnlZ.H \ df_star_variance') ./ ...
      (3 * f_star_variance);

  % approximate predictive distribution for y* (observations)
  y_star_mean     = f_star_mean;
  y_star_variance = f_star_variance + noise_variance;

  % log predictive probabilities
  if (nargin > 8)
    log_probabilities = likGauss(hyperparameters.lik, y_star, ...
                                 f_star_mean, f_star_variance, 'infEP');
  end

end

% returns diag(AB)
function result = product_diag(A, B)

  result = sum(B .* A')';

end