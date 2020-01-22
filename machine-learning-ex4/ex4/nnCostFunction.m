function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1

a1 = X;
a1 = [ones(rows(a1), 1) a1];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(rows(a2), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hyp = a3; % 5000 x 10

y_recoded = [1:num_labels] == y; % 5000 x 10

J_aux = -y_recoded .* log(hyp) - (1 - y_recoded) .* log(1 - hyp) % 5000 x 10

J = (1 / m) * sum(J_aux(:));

% Part 3 - for Cost Function

Theta1_without_bias_squared = Theta1(:, 2:end).^2;
Theta2_without_bias_squared = Theta2(:, 2:end).^2;

reg_term = (lambda / (2 * m)) * (sum(Theta1_without_bias_squared(:)) + sum(Theta2_without_bias_squared(:)));

J = J + reg_term;

% Part 2

deltaM_1 = zeros(size(Theta1));
deltaM_2 = zeros(size(Theta2));

for t = 1:m,
    a_1 = X(t, :); % 1 x 400
    a_1 = [1 a_1]; % 1 x 401
    
    z_2 = a_1 * Theta1'; % 1 x 25
    a_2 = sigmoid(z_2);
    a_2 = [1 a_2]; % 1 x 26

    z_3 = a_2 * Theta2'; % 1 x 10
    a_3 = sigmoid(z_3);
    
    delta_3 = a_3 - y_recoded(t, :); % 1 x 10
    
    delta_2 = (delta_3 * Theta2(:, 2:end)) .* sigmoidGradient(z_2); % 1 x 10 * 10 x 25 = 1 x 25
    
    deltaM_2 = deltaM_2 + delta_3' * a_2;
    deltaM_1 = deltaM_1 + delta_2' * a_1;
    
    Theta2_grad = (1 / m) * (deltaM_2);
    Theta1_grad = (1 / m) * (deltaM_1);
    
endfor

% Part 3 - for Gradients

reg_term_theta2 = zeros(size(Theta2_grad));
reg_term_theta1 = zeros(size(Theta1_grad));

reg_term_theta2(:, 2:end) = (lambda / m) * Theta2(:, 2:end);
reg_term_theta1(:, 2:end) = (lambda / m) * Theta1(:, 2:end);

Theta2_grad = Theta2_grad + reg_term_theta2;
Theta1_grad = Theta1_grad + reg_term_theta1;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
