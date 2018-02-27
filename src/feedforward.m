function [y,y_dot] = feedforward(x,W,b,l)
%FEEDFORWARD    returns the activation value of each nodes after neural
%network forward propagation
%   [y,y_dot] = FEEDFORWARD(x,W,b,l) returns y, y_dot representing the
%   activation value of each network nodes and the corresponding
%   derivative, computed using sigmoid function and the derivative of the
%   sigmoid function respectively. Here W is the weights, b is the biases,
%   and l is the number of network's layers.
%
%   see also: sigmoid, sigmoidPrime
    
    y = x;
    y_dot = cell(1,l-1);

    for i = 1:l-1
        y{i+1} = sigmoid(W{i}*y{i}+b{i});
        y_dot{i} = sigmoidPrime(W{i}*y{i}+b{i});
    end
end