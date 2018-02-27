function [W,b] = updateWeightBias(X,Y,eta,W,b,n,l)
%UPDATEWEIGHTBIAS   update the value of the weights and biases of the network
%using gradient decent and back propagation
%   [W,b] = UPDATEWEIGHTBIAS(X,Y,eta,W,b,n,l) computes the new value of the
%   weight W and biases b of a network with l layers using matrix X as a
%   set of network inputs and matrix Y as a set of the desired network
%   output. SIZE(X,1), SIZE(Y,1) is the number of nodes at the input and
%   output layer respectively, while SIZE(X,2), SIZE(Y,2) is the number of
%   the training data. Here, n stores the number of nodes in each network's
%   layer and eta is the learning rate.
%
%   see also: backprop, gradientDecent

    % initialize the gradient of the weighting matrices and biases
    nabla_W = cell([1,l-1]);
    nabla_b = cell([1,l-1]);
    for i = 1:l-1
        nabla_W{i} = zeros(n(i+1),n(i));
        nabla_b{i} = zeros(n(i+1),1);
    end

    % compute the gradient of the weights and biases using back
    % progpagation
    for i = 1:size(X,2)
        [dnabla_W, dnabla_b] = backprop(X(:,i),Y(:,i),W,b,l);
        for j = 1:l-1
            nabla_W{j} = nabla_W{j}+dnabla_W{j};
            nabla_b{j} = nabla_b{j}+dnabla_b{j};
        end
    end
        
    % update the value of weight and bias
    for i = 1:l-1
        W{i} = W{i} - (eta/size(X,2)).*nabla_W{i};
        b{i} = b{i} - (eta/size(X,2)).*nabla_b{i};
    end
end