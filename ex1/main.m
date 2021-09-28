data = load('ex1data1.txt');
X = data(:,1); y = data(:,2);
m = length(X);
X = [ones(m,1), data(:,1)];

theta = zeros(2,1);
iterations = 1500;
alpha = 0.01;


data = load('ex1data2.txt');
X = data(:,1:2);
[X, mu, sigma] = featureNormalize(X);
X = [ones(m, 1) X];
y = data(:,3);
m = length(y);

fprintf('x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))