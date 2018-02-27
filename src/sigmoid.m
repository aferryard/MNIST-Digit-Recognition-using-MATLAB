function y = sigmoid(X)
%SIGMOID    computes the value of the neutron using the sigmoid function
%   y = SIGMOID(X) returns the sigmoid function of the elements of vector
%   X, i.e. y = 1./(1+exp(-X))
%
%   see also: exp

y = 1./(1+exp(-X));
end