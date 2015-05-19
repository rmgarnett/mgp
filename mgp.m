% MGP approximates marginal GP predictions.
%
% Suppose we have a Gaussian process model on a latent function f:
%
%   p(f | \theta) = GP(f; \mu(x; \theta), K(x, x'; \theta)),
%
% where \theta are the hyperparameters of the model. Suppose we have a
% dataset D = (X, y) of observations and a test point x*. This
% function returns the mean and variance of the approximate marginal
% predictive distributions for the associated observation value y* and
% latent function value f*:
%
%   p(y* | x*, D) = \int p(y* | x*, D, \theta) p(\theta | D) d\theta,
%   p(f* | x*, D) = \int p(f* | x*, D, \theta) p(\theta | D) d\theta,
%
% where we have marginalized over the hyperparameters \theta. The
% approximate posterior is derived using he "MGP" approximation
% described in
%
%   Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
%   Embeddings for Gaussian Processes. (2014). 30th Conference on
%   Uncertainty in Artificial Intelligence (UAI 2014).
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
% which must also be in your MATLAB path.
%
% Notes
% -----
%
% The approximation relies first on a Gaussian approximation to the
% hyperparameter posterior p(\theta | D):
%
%   p(\theta | D) ~ N(\hat{\theta}, \Sigma).
%
% The hyperparameter vector passed in here is assumed to represent the
% mean of this approximate posterior, \hat{\theta}. The user may, if
% desired, also specify the covariance or precision matrix of the
% approximate posterior. See "Specifiying Covariance" and "Specifying
% Precision" below for details.
%
% If not specified, the code derives an approximate posterior by
% performing a Laplace approximation at the given value of
% \hat{\theta}. See "Laplace Approximation" below for details.
%
% Specifying Covariance
% ---------------------
%
% To specify the covariance matrix \Sigma of the approximate
% hyperparameter posterior distribution, augment the GPML
% hyperparameter struct theta (representing the approximate posterior
% mean \hat{\theta}) with a field .Sigma, containing a struct with
% fields:
%
%   theta.Sigma.value
%   theta.Sigma.covariance_ind
%   theta.Sigma.likelihood_ind
%   theta.Sigma.mean_ind
%
% theta.Sigma.value should be a (m x m) matrix containing the desired
% covariance matrix. The remaining fields index into Sigma.value
% identifying rows/columns with different hyperparameter subsets. Note
% that this is the same structure as used for specifying the
% hyperparameter Hessian matrix HnlZ in the gpml_extensions
% repository; see hessians.m in that repository for more information.
%
% Specifying Precision
% --------------------
%
% To specify the precision matrix \Sigma^{-1} of the approximate
% hyperparameter posterior distribution, augment the GPML
% hyperparameter struct theta (representing the approximate posterior
% mean \hat{\theta}) with a field .Sigma_inv. See the "Specifying
% Covariance" section above for the expected format of this field,
% noting that of course in this case theta.Sigma_inv.value will
% contain the desired (m x m) hyperparameter precision matrix.
%
% Laplace Approximation
% ---------------------
%
% If the user does not specify the covariance or precision matrix of
% the approximate hyperparameter posterior distribution, then a
% Laplace approximation will be performed.
%
% In this case, this code requires that the provided hyperparameters
% be the MLE hyperparameters:
%
%   \hat{theta} = argmax_\theta log p(y | X, \theta),
%
% or, if using a hyperparameter prior p(\theta), the MAP
% hyperparameters:
%
%   \hat{theta} = argmax_\theta log p(y | X, \theta) + log p(\theta).
%
% In this case, this function will not perform the maximization over
% \theta but will rather assume that the given hyperparameters
% represent the MLE/MAP point.
%
% Usage
% -----
%
% The usage of mgp.m is identical to the gp(...) function from the
% GPML toolkit in prediction mode:
%
%  [y_star_mean, y_star_variance,                                ...
%   f_star_mean, f_star_variance,                                ...
%   log_probabilities, posterior,                                ...
%   y_star_mean_gp, y_star_variance_gp,                          ...
%   f_star_variance_gp, log_probabilities_gp] =                  ...
%      mgp(theta, inference_method, mean_function, covariance_function, ...
%          likelihood, x, y, x_star, y_star)
%
% There are four additional output arguments: y_star_mean_gp,
% y_star_variance_gp, f_star_variance_gp, and
% log_probabilities_gp. These provide:
%
%     E[y* | x*, D, \hat{\theta}],
%   Var[y* | x*, D, \hat{\theta}],
%   Var[f* | x*, D, \hat{\theta}],
%
% and
%
%   log p(y* | x*, D, \hat{theta}),
%
% respectively. That is, these give the outputs that would have been
% provided by calling gp(...) without the MGP corrections applied, if
% desired. Note that under the MGP approximation, f_star_mean equals
% what would be the (nonexistent, redundant) f_star_mean_gp output.
%
% Inputs
% ------
%
%                 theta: a GPML hyperparameter struct containing the
%                        mean of the approximate hyperparameter
%                        posterior (and possibly the
%                        covariance/precision of this approximation,
%                        see above)
%      inference_method: a GPML inference method
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood
%                     x: training observation locations (n x D)
%                     y: training observation values (n x 1) or
%                        GPML posterior struct
%                x_star: test observation locations (n_star x D)
%                y_star: test observation values (n_star x 1) (optional)
%
% Outputs
% -------
%
%            y_star_mean: the approximate   E[y* | x*, D]
%        y_star_variance: the approximate Var[y* | x*, D]
%            f_star_mean: the approximate   E[f* | x*, D]
%        f_star_variance: the approximate Var[f* | x*, D]
%      log_probabilities: if test observation values y_star are
%                         provided, a vector containing the
%                         approximate log predictive probabilities
%
%                           log p(y* | x*, D).
%
%              posterior: a GPML posterior struct corresponding to the
%                         provided training data and MLE/MAP
%                         hyperparameters
%
% Outputs conditioned on \hat{\theta} (no MGP corrections applied,
% matches outputs from gp.m):
%
%         y_star_mean_gp:   E[y* | x*, D, \hat{\theta}]
%     y_star_variance_gp: Var[y* | x*, D, \hat{\theta}]
%     f_star_variance_gp: Var[f* | x*, D, \hat{\theta}]
%   log_probabilities_gp: if test observation values y_star are
%                         provided, a vector containing the
%                         approximate log predictive probabilities
%
%                           log p(y* | x*, D, \hat{\theta}).
%
% See also GP.

% Copyright (c) 2014--2015 Roman Garnett.

function [y_star_mean, y_star_variance,                                ...
          f_star_mean, f_star_variance,                                ...
          log_probabilities, posterior,                                ...
          y_star_mean_gp, y_star_variance_gp,                          ...
          f_star_variance_gp, log_probabilities_gp] =                  ...
      mgp(theta, inference_method, mean_function, covariance_function, ...
          likelihood, x, y, x_star, y_star)

  % perform initial argument checks/transformations
  [theta, inference_method, mean_function, covariance_function, likelihood] ...
      = check_arguments(theta, inference_method, mean_function, ...
          covariance_function, likelihood, x);

  % convenience handles
  mu        = @(varargin) feval(mean_function{:},       theta.mean, varargin{:});
  K         = @(varargin) feval(covariance_function{:}, theta.cov,  varargin{:});
  inference = @()         feval(inference_method{:},    theta, ...
          mean_function, covariance_function, likelihood, x, y);

  % S will contain either Sigma or Sigma_inv, and Sigma_times will
  % provide a handle for computing \Sigma * x

  have_posterior = false;
  if (isfield(theta, 'Sigma'))
    have_posterior = true;

    S = theta.Sigma;
    Sigma_times = @(x) (S.value * x);
  elseif (isfield(theta, 'Sigma_inv'))
    have_posterior = true;

    S = theta.Sigma_inv;
    Sigma_times = @(x) (S.value \ x);
  end

  % find GP posterior at the approximate posterior mean \theta, as well
  % as the derivatives of alpha and diag W^{-1} with respect to
  % \theta

  if (have_posterior)
    [posterior, ~, ~, dalpha, dWinv] = inference();
  else
    % neither covariance nor precision matrix given, perform Laplace
    % approximation

    % set S to Hessian of log likelihood/posterior
    [posterior, ~, ~, dalpha, dWinv, S] = inference();

    Sigma_times = @(x) (S.value \ x);
  end

  % find the predictive distribution conditioned on \theta
  if ((nargin > 8) && (nargout > 8) && ~isempty(y_star))
    % log probaiblities requested

    [y_star_mean_gp, y_star_variance_gp, f_star_mean, f_star_variance_gp, ...
     log_probabilities_gp] = gp(theta, inference_method, mean_function, ...
            covariance_function, likelihood, x, posterior, x_star, y_star);

  else
    [y_star_mean_gp, y_star_variance_gp, f_star_mean, f_star_variance_gp] ...
        = gp(theta, inference_method, mean_function, covariance_function, ...
             likelihood, x, posterior, x_star);

    log_probabilities_gp = [];
  end

  % precompute k*' [ K + W^{-1} ]^{-1}; it's used a lot
  k_star = K(x, x_star);

  % handle different posterior parameterizations
  if (is_chol(posterior.L))
    % posterior.L contains chol(I + W^{1/2} K W^{1/2})

    % [ K + W^{-1} ]^{-1} =
    % W^{1/2} [I + W^{1/2} K W^{1/2} ]^{-1} W^{1/2}
    k_star_V_inv = ...
        bsxfun(@times, posterior.sW', ...
               solve_chol(posterior.L, bsxfun(@times, posterior.sW, k_star))');
  else
    % posterior.L contains -[ K + W^{-1} ]^{-1}

    k_star_V_inv = -k_star' * posterior.L;
  end

  % for each hyperparameter \theta_i, we must compute:
  %
  %   d mu_{f | D}(x*; \theta) / d \theta_i, and
  %   d  V_{f | D}(x*; \theta) / d \theta_i.

  num_test = size(x_star, 1);
  num_hyperparameters = size(S.value, 1);

  df_star_mean     = zeros(num_test, num_hyperparameters);
  df_star_variance = zeros(num_test, num_hyperparameters);

  % partials with respect to covariance hyperparameters
  for i = 1:numel(S.covariance_ind)
    dK           = K(x,      [],     i);
    dk_star      = K(x,      x_star, i);
    dk_star_star = K(x_star, 'diag', i);

    df_star_mean(:, S.covariance_ind(i)) = ...
        dk_star' * posterior.alpha + k_star' * dalpha.cov(:, i);

    df_star_variance(:, S.covariance_ind(i)) = ...
        dk_star_star - ...
        product_diag(k_star_V_inv, ...
                     2 * dk_star - (dK + diag(dWinv.cov(:, i))) * k_star_V_inv');
  end

  % partials with respect to likelihood hyperparameters
  for i = 1:numel(S.likelihood_ind)
    df_star_mean(:, S.likelihood_ind(i)) = k_star' * dalpha.lik(:, i);

    df_star_variance(:, S.likelihood_ind(i)) = ...
        product_diag(k_star_V_inv, ...
                     bsxfun(@times, dWinv.lik(:, i), k_star_V_inv'));
  end

  % partials with respect to mean hyperparameters
  for i = 1:numel(S.mean_ind)
    dm_star = mu(x_star, i);

    df_star_mean(:, S.mean_ind(i)) = dm_star + k_star' * dalpha.mean(:, i);

    df_star_variance(:, S.mean_ind(i)) = ...
        product_diag(k_star_V_inv, ...
            bsxfun(@times, dWinv.mean(:, i), k_star_V_inv'));
  end

  % the MGP approximation inflates the predictive variance to
  % account for uncertainty in \theta
  f_star_variance = ...
      (4 / 3) * f_star_variance_gp + ...
      product_diag(df_star_mean,     Sigma_times(df_star_mean')) + ...
      product_diag(df_star_variance, Sigma_times(df_star_variance')) ./ ...
      (3 * f_star_variance_gp);

  % approximate predictive distribution for y* (observations). if y*
  % given, compute log predictive probabilities.
  if ((nargin > 8) && (nargout > 4) && ~isempty(y_star))
    [log_probabilities, y_star_mean, y_star_variance] = ...
        feval(likelihood{:}, theta.lik, y_star, f_star_mean, f_star_variance);
  else
    [~,                 y_star_mean, y_star_variance] = ...
        feval(likelihood{:}, theta.lik, [],     f_star_mean, f_star_variance);

    log_probabilities = [];
  end

end

% returns diag(AB) without computing the full product
function result = product_diag(A, B)

  result = sum(B .* A')';

end