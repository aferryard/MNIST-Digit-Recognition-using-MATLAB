function [dnabla_W,dnabla_b] = backprop(X,Y,W,b,l)
%BACKPROP returns the gradient of the weights and biases
%   [dnabla_W,dnabla_b] = BACKPROP(X,Y,W,b,l) returns dnabla_W and dnabla_b
%   ,representing the gradient of weights W and biases b respectively, of a
%   network with l layers using vector X as the network's inputs and vector
%   Y as the desired network's outputs.
%   
%   see also: feedforward

    % fill in the input nodes
    a{1} = X;
    
    % fill in the hidden layer nodes up to the output layers
    [a,a_dot] = feedforward(a,W,b,l);
    
    % gradient descent
    delta = (a{end}-Y).*a_dot{end};
    
    dnabla_W{l-1} = delta*a{l-1}';
    dnabla_b{l-1} = delta;
    
    for i = l-2:-1:1
        delta = W{i+1}'*delta.*a_dot{i};
        dnabla_W{i} = delta*a{i}';
        dnabla_b{i} = delta;
    end
end
