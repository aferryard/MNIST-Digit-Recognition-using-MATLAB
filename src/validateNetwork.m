function correct = validateNetwork(X,Y,W,b,l)
%VALIDATENETWORK   returns the number of test inputs for which the neural
%network outputs the correct results
%   correct = VALIDATENETWORK(X,Y,W,b,l) compares the output of a network
%   with a 1-by-n vector Y representing the desired output and returns the
%   number of the correct values. Here, W is the network's weights, b is
%   the biases, l is the number of layers, and X is a m-by-n matrix
%   representing the network's input where m is the number of first layer's
%   nodes and n is the number of test data.
%
%   see also: feedforward, max.
    
    % initialize the function output
    correct = 0;
    
    for i = 1:size(Y,2)
        % fill in the input layers
        a{1} = X(:,i);
        
        % fill in the hidden layer nodes up to the output layers
        [a,~] = feedforward(a,W,b,l);
    
        % find the maximum index value of the output layer
        [~,idx] = max(a{l});
    
        if idx-1 == Y(i)
            correct = correct + 1;
        end
    end  
end