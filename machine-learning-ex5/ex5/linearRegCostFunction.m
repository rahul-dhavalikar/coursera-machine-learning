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

h = (X * theta); %mx2 * 2x1 = mx1
temp = (h - y) ;  
cost = temp' * temp;
J = cost/(2*m);

theta2 = theta(2:end);  % skipping theta(1)
regularization_term = (lambda / (2*m)) * (theta2' * theta2);
J = J + regularization_term;

% grad
grad = (X' * (h - y)) / m;    % unregularized grad
theta3 = [0; theta2];         % assiging theta(1) as 0
regularization_term = (lambda / m) * theta3;
grad = grad + regularization_term;












% =========================================================================

grad = grad(:);

end
