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
a1 = [ones(m,1) X]; % 5000*401
z2 = a1 * Theta1'; % 5000*25
a2 = [ones(m,1) sigmoid(z2)]; % 5000*26
z3 = a2 * Theta2'; % 5000*10
a3 = sigmoid(z3); % 5000*10

index = zeros(m, max(y));
idx = sub2ind(size(index), 1:m, y(:)');
index(idx) = 1;

reg = lambda/(2*m)*(sum(Theta1.^2, 'all')+sum(Theta2.^2, 'all'));
J = sum(log(a3).*index + log(1-a3).*(1-index), 'all')/(-m) + reg;

delta3 = a3 - index; % 5000*10
delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(z2); % 5000*25
LD2_1 = delta3'*a2(:,1); % 10*1
LD2_rest = delta3'*a2(:,2:end) + lambda*Theta2(:,2:end); %10*25
LD1_1 = delta2'*a1(:,1); % 25*1
LD1_rest = delta2'*a1(:,2:end) + lambda*Theta1(:,2:end); % 25*400


Theta1_grad = [LD1_1 LD1_rest]/m;
Theta2_grad = [LD2_1 LD2_rest]/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end