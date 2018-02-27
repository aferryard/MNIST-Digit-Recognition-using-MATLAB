function y = sigmoidPrime(X)
%SIGMOIDPRIME   computes the derivative of the sigmoid function
%   y = SIGMOIDPRIME(X) returns the sigmoid function derivative of the
%   elements of vector X.
%
%   see also: sigmoid

y = sigmoid(X).*(1-sigmoid(X));
end