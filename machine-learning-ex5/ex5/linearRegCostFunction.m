function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% COST

pred = X * theta;

sqrErrors = (pred - y) .^ 2;

regTerm = lambda / (2 * m) * sum(theta(2:end) .^ 2)

J = 1 / (2 * m) * sum(sqrErrors) + regTerm;

% GRAD

grad_1 = (1 / m) * sum((pred - y) .* X(:, 1));

% SUM FOR THREE VARIABLES = NUM NUM NUM (YOU NEED TO TRANSPOSE)
grad_2end = (1 / m) * sum((pred - y) .* X(:, 2:end))' + (lambda / m) * theta(2:end);

grad = [grad_1; grad_2end(:)];


% =========================================================================

grad = grad(:);

end
