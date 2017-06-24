function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); % h = mx1

temp = (-y .* log(h)) - ((1 .- y) .* log(1 .- h));
J = sum(temp) / m;      % J not regularized
theta2 = theta(2:end);  % skipping theta(1)
regularization_term = (lambda / (2*m)) * (theta2'*theta2);
J = J + regularization_term;

grad = (X' * (h - y)) / m;    % old grad
theta3 = [0; theta2];         % skipping theta(1)
regularization_term = (lambda / m) * theta3;
grad = grad + regularization_term;



% =============================================================

end
