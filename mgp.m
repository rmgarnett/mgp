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
% Notes
% -----
%
% This code is only appropriate for GP regression! Exact inference
% with a Gaussian observation likelihood is assumed.
%
% The MGP approximation requires that the provided hyperparameters be
% the MLE hyperparameters:
%
%   \hat{theta} = argmax_\theta log p(y | X, \theta),
%
% or, if using a hyperparameter prior p(\theta), the MAP
% hyperparameters:
%
%   \hat{theta} = argmax_\theta log p(y | X, \theta) + log p(\theta).
%
% This function does not perform the maximization over \theta but
% rather assumes that the given hyperparameters represent \hat{theta}.
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
% Usage
% -----
%
% The usage of mgp.m is identical to the gp(...) function from the
% GPML toolkit in prediction mode:
%
%   [y_star_mean, y_star_variance, ...
%    f_star_mean, f_star_variance, ...
%    log_probabilities, posterior, ...
%    y_star_variance_gp, f_star_variance_gp, ...
%    log_probabilities_gp] = ...
%        mgp(hyperparameters, inference_method, mean_function, ...
%            covariance_function, likelihood, x, y, x_star, y_star);
%
% There are three additional output arguments: y_star_variance_gp,
% f_star_variance_gp, and log_probabilities_gp. These provide:
%
%   Var[y* | x*, D, \hat{\theta}],
%   Var[f* | x*, D, \hat{\theta}],
%
% and
%
%   log p(y* | x*, D, \hat{theta}),
%
% respectively. That is, these give the outputs that would have been
% provided by calling gp(...) without the MGP corrections applied,
% if desired.
%
% Inputs
% ------
%
%       hyperparameters: a GPML hyperparameter struct containing the
%                        MLE/MAP hyperparameters
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
%     y_star_variance_gp: Var[y* | x*, D, \hat{\theta}]
%     f_star_variance_gp: Var[f* | x*, D, \hat{\theta}]
%   log_probabilities_gp: if test observation values y_star are
%                         provided, a vector containing the
%                         approximate log predictive probabilities
%
%                           log p(y* | x*, D, \hat{\theta}).
%
% See also GP.

% Copyright (c) 2014 Roman Garnett.

function [y_star_mean, y_star_variance, ...
          f_star_mean, f_star_variance, ...
          log_probabilities, posterior, ...
          y_star_variance_gp, f_star_variance_gp, ...
          log_probabilities_gp] = ...
      mgp(hyperparameters, inference_method, mean_function, ...
          covariance_function, likelihood, x, y, x_star, y_star)

  % perform initial argument checks/transformations
  [hyperparameters, inference_method, mean_function, covariance_function, ...
   likelihood] = check_arguments(hyperparameters, inference_method, ...
          mean_function, covariance_function, likelihood, x);

  % convenience handles
  mu = @(varargin) feval(mean_function{:},       hyperparameters.mean, varargin{:});
  K  = @(varargin) feval(covariance_function{:}, hyperparameters.cov,  varargin{:});

  % find GP posterior and Hessian of negative log likelihood evaluated
  % at the MLE/MAP \theta, as well as the derivatives of alpha and
  % diag W^{-1} with respect to \theta
  [posterior, ~, ~, HnlZ, dalpha, dWinv] = inference_method(hyperparameters, ...
          mean_function, covariance_function, likelihood, x, y);

  % find the predictive distribution conditioned on the MLE/MAP \theta
  if ((nargin > 8) && (nargout > 8) && ~isempty(y_star))
    % log probaiblities conditioned on MLE/MAP \theta requested
    [y_star_mean, y_star_variance_gp, f_star_mean, f_star_variance_gp, ...
     log_probabilities_gp] = gp(hyperparameters, inference_method, ...
            mean_function, covariance_function, likelihood, x, posterior, ...
            x_star, y_star);
  else
    [y_star_mean, y_star_variance_gp, f_star_mean, f_star_variance_gp] ...
        = gp(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, x, posterior, x_star);

    log_probabilities_gp = [];
  end

  % precompute k*' [ K + W^{-1} ]^{-1}; it's used a lot
  k_star = K(x, x_star);

  noise_variance = exp(2 * hyperparameters.lik);

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

  num_test            = size(x_star, 1);
  num_hyperparameters = size(HnlZ.H, 1);

  df_star_mean     = zeros(num_test, num_hyperparameters);
  df_star_variance = zeros(num_test, num_hyperparameters);

  % partials with respect to covariance hyperparameters
  for i = 1:numel(HnlZ.covariance_ind)
    dK           = K(x,      [],     i);
    dk_star      = K(x,      x_star, i);
    dk_star_star = K(x_star, 'diag', i);

    df_star_mean(:, HnlZ.covariance_ind(i)) = ...
        dk_star' * posterior.alpha + k_star' * dalpha.cov(:, i);

    df_star_variance(:, HnlZ.covariance_ind(i)) = ...
        dk_star_star - ...
        product_diag(k_star_V_inv, ...
                     2 * dk_star - (dK + diag(dWinv.cov(:, i))) * k_star_V_inv');
  end

  % partials with respect to likelihood hyperparameters

  for i = 1:numel(HnlZ.likelihood_ind)
    df_star_mean(:, HnlZ.likelihood_ind(i)) = k_star' * dalpha.lik(:, i);

    df_star_variance(:, HnlZ.likelihood_ind(i)) = ...
        product_diag(k_star_V_inv, ...
                     bsxfun(@times, dWinv.lik(:, i), k_star_V_inv'));
  end

  % partials with respect to mean hyperparameters
  for i = 1:numel(HnlZ.mean_ind)
    dm_star = mu(x_star, i);

    df_star_mean(:, HnlZ.mean_ind(i)) = dm_star + k_star' * dalpha.mean(:, i);

    df_star_variance(:, HnlZ.mean_ind(i)) = ...
        product_diag(k_star_V_inv, ...
            bsxfun(@times, dWinv.mean(:, i), k_star_V_inv'));
  end

  % the MGP approximation inflates the predictive variance to
  % account for uncertainty in \theta
  f_star_variance = ...
      (4 / 3) * f_star_variance_gp + ...
      product_diag(df_star_mean,     HnlZ.H \ df_star_mean') + ...
      product_diag(df_star_variance, HnlZ.H \ df_star_variance') ./ ...
      (3 * f_star_variance_gp);

  % approximate predictive distribution for y* (observations)
  y_star_variance = f_star_variance + noise_variance;

  % if y* given, compute log predictive probabilities
  if ((nargin > 8) && (nargout > 4) && ~isempty(y_star))
    log_probabilities = feval(likelihood{:}, hyperparameters.lik, y_star, ...
            f_star_mean, f_star_variance);
  else
    log_probabilities = [];
  end

end

% returns diag(AB) without computing the full product
function result = product_diag(A, B)

  result = sum(B .* A')';

end

% performs argument checks/transformations similar to those found in
% gp.m from GPML but guaranteed to be compatible with the MGP
function [hyperparameters, inference_method, ...
          mean_function, covariance_function, likelihood] = ...
      check_arguments(hyperparameters, inference_method, ...
                      mean_function, covariance_function, ...
                      likelihood, x)

  % default to exact inference
  if (isempty(inference_method))
    inference_method = @exact_inference;
  end

  % default to zero mean function
  if (isempty(mean_function))
    mean_function = {@zero_mean};
  end

  % no default covariance function
  if (isempty(covariance_function))
    error('mgp:missing_argument', ...
          'covariance function must be defined!');
  end

  % default to Gaussian likelihood
  if (isempty(likelihood))
    likelihood = {@likGauss};
  end

  % allow string/function handle input for mean, covariance, and
  % likelihood functions; convert to cell arrays if necessary
  if (ischar(mean_function) || ...
      isa(mean_function, 'function_handle'))
    mean_function = {mean_function};
  end

  if (ischar(covariance_function) || ...
      isa(covariance_function, 'function_handle'))
    covariance_function = {covariance_function};
  end

  if (ischar(likelihood) || ...
      isa(likelihood, 'function_handle'))
    likelihood = {likelihood};
  end

  % ensure all hyperparameter fields exist
  for field = {'cov', 'lik', 'mean'}
    if (~isfield(hyperparameters, field{:}))
      hyperparameters.(field{:}) = [];
    end
  end

  % check dimension of hyperparameter fields
  D = size(x, 2);

  expression = feval(mean_function{:});
  if (numel(hyperparameters.mean) ~= eval(expression))
    error('mgp:incorrect_specification', ...
          'wrong number of mean hyperparameters! (%i given, %s expected)', ...
          numel(hyperparameters.mean), ...
          expression);
  end

  expression = feval(covariance_function{:});
  if (numel(hyperparameters.cov) ~= eval(expression))
    error('mgp:incorrect_specification', ...
          'wrong number of covariance hyperparameters! (%i given, %s expected)', ...
          numel(hyperparameters.cov), ...
          expression);
  end

  if (numel(hyperparameters.lik) ~= 1)
    error('mgp:incorrect_specification', ...
          'wrong number of likelihood hyperparameters! (%i given, 1 expected)', ...
          numel(hyperparameters.lik));
  end

  % if infExact specified, use drop-in replacement exact_inference
  % instead
  if (isequal(inference_method, @infExact))
    inference_method = @exact_inference;
  end

end