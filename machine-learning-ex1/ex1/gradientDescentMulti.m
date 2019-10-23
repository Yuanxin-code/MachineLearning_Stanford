function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    for i = 1:size(X,1)
        h = theta(1,1) + theta(2,1) * X(i,2) + theta(2,1) * X(i,3);
        sum1 = sum1 + (h-y(i,1));
        sum2 = sum2 + (h-y(i,1)) * X(i,2);
        sum3 = sum3 + (h-y(i,1)) * X(i,3);
    end
    theta(1,1) = theta(1,1) - (alpha / size(X,1)) * sum1;
    theta(2,1) = theta(2,1) - (alpha / size(X,1)) * sum2;
    theta(3,1) = theta(3,1) - (alpha / size(X,1)) * sum3;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
