function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

temp = X * theta; %mx2 * 2x1 = mx1
temp = temp - y;  
temp = temp .^ 2; 
cost = sum(temp); %1x1
J = cost/(2*m);




% =========================================================================

end
