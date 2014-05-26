% generate fake data
f = @(x) (sin(x) ./ x + 0.1 * randn(size(x)));

x = 15 * randn(20, 1);
y = f(x);

% setup GP

% We'll use a zero mean and a squared exponential covariance. We must
% use mean/covariance functions that support an extended GPML syntax
% allowing the calculation of second partial dervivatives with respect
% to hyperparameters. The gpml_extensions package contains
% implementations of some common choices.

inference_method    = @exact_inference;
mean_function       = {@zero_mean};
covariance_function = {@isotropic_sqdexp_covariance};

% initial hyperparameters
hyperparameters.cov  = [log(1); log(1)];
hyperparameters.lik  = log(0.1);
hyperparameters.mean = [];

% find MLE hyperparameters
mle_hyperparameters = minimize(hyperparameters, @gp, 20, inference_method, ...
        mean_function, covariance_function, [], x, y);

% get predictions from GP conditioned on MLE hyperparameters
x_star = linspace(-30, 30, 500)';
y_star = f(x_star);

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(mle_hyperparameters, inference_method, mean_function, ...
       covariance_function, [], x, y, x_star, y_star);

fprintf(' GP/MLE: E[log p(y* | x*, D)] = %0.3f\n', ...
        mean(log_probabilities));

% plot GP predictions
figure(1);
set(gcf, 'color', 'white');
subplot(2, 2, 1);
plot_predictions;

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    mgp(mle_hyperparameters, inference_method, mean_function, ...
        covariance_function, [], x, y, x_star, y_star);

fprintf('MGP/MLE: E[log p(y* | x*, D)] = %0.3f\n', ...
        mean(log_probabilities));

% plot MGP predictions
subplot(2, 2, 2);
plot_predictions;

% add hyperparameter priors
priors.cov  = ...
    {get_prior(@gaussian_prior, log(1), 0.1), ...
     get_prior(@gaussian_prior, log(1), 0.1)};
priors.lik  = ...
    {get_prior(@gaussian_prior, log(0.01), 0.1)};
priors.mean = {};

prior = get_prior(@independent_prior, priors);

inference_method = add_prior_to_inference_method(inference_method, prior);

% find MAP hyperparameters
map_hyperparameters = minimize(hyperparameters, @gp, 20, inference_method, ...
        mean_function, covariance_function, [], x, y);

% get predictions from GP conditioned on MAP hyperparameters
[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(map_hyperparameters, inference_method, mean_function, ...
       covariance_function, [], x, y, x_star, y_star);

fprintf(' GP/MAP: E[log p(y* | x*, D)] = %0.3f\n', ...
        mean(log_probabilities));

% plot GP predictions
subplot(2, 2, 3);
plot_predictions;

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    mgp(map_hyperparameters, inference_method, mean_function, ...
        covariance_function, [], x, y, x_star, y_star);

fprintf('MGP/MAP: E[log p(y* | x*, D)] = %0.3f\n', ...
        mean(log_probabilities));

% plot MGP predictions
subplot(2, 2, 4);
plot_predictions;