%% Digit Recognition Using Neural Networks
% This is the neural network implementation of handwritten
% digit recognition based on Michael Nielsen's book:
% <http://neuralnetworksanddeeplearning.com/index.html? Neural
% Networks and Deep Learning> Chapter 1. This matlab code is a modified
% version of his python code which can be found
% <https://github.com/mnielsen/neural-networks-and-deep-learning here>.

%% Initializations
% Close all figures, clear workspace, and clear the command window.

close all
clear all
clc

%% Initialize Parameters
% Several parameters that need to be initialized includes: the number of
% nodes in each hidden layer, the number of training epochs, the number of
% training data in each mini batch, and the learning rate.

nh = [30];                                                      % one hidden layer with 30 nodes
epoch = 1;                                                      % training epochs
mini_batch_size = 10;                                           % mini bacth size
eta = 3.0;                                                      % learning rate

%% Load the MNIST data
% The provided MNIST data is saved in a zip file as a python _pickle_
% object. In order to use it in Matlab, the data is converted into |.mat|
% file. In this case, the MNIST data has been saved as a |.mat| file.

load('mnist.mat');

%% 
% Next, the data is divided into training data (to trains the
% network), test data (to verify the network accuracy), and validation data
% (not used in this code). Each of them contains the inputs (pixel data)
% and the results (integer corresponds to the digit). The training results
% needs to be converted into 10-dimensional vector where each vector row
% corresponds to the activation value of each digit [0, 1, ..., 9].

training_inputs = double(mnist{1,1}');                          % transpose the matrix to make the pixel data as the row element
training_results = vectorizeData(mnist{1,2});                   % convert the digit into the activation value of the neural network

validation_inputs = double(mnist{2,1}');
validation_results = mnist{2,2};

test_inputs = double(mnist{3,1}');
test_results = mnist{3,2};

%% Create the Network
% Next, The network is created. It contains |l = 1+length(nh)+1|
% layers where the first layer is the input from MNIST data and the last
% layer is the output. 

l  = 1+length(nh)+1;                                            % number of layer     

%% 
% Each node in the input layer correspons to each pixel of the image and
% each node in the output layer corresponds to each digit [0, 1, ..., 9].
          
n = zeros(l,1);                                                 % initialize the nodes
n(1) = size(mnist{1,1},2);                                      % nodes at the input layer, number of pixel square

for i = 1:length(nh)
    n(1+i) = nh(i);                                             % nodes at the hidden layers
end

n(l) = 10;                                                      % nodes at the output layer, number of digit

%% Initialize the Weights and Biases
% The weights and biases are generated randomly using a random number
% generator. The random number is normally distributed with mean 0 and
% variance 1.

rng(0,'twister');                                               % Initialize the random number generator

W = cell([1,l-1]);                                              % Weight
b = cell([1,l-1]);                                              % Bias

for i = 1:l-1
    W{i} = randn(n(i+1),n(i));
    b{i} = randn(n(i+1),1);
end

%% Main Loop
% Here is where the training process happens. The network is
% trained using a stochastic gradient descent and backpropagation method.

corr_val = zeros(epoch,1);                                      % number of correct output for each epoch

[~,col] = size(training_inputs);                                % access the number of column

%% 
% In each epoch, the optimal weights and biases are
% computed. Then network performance is evaulated using a set of test data.

for i = 1:epoch
    
    %%
    % First, the training data and training results are randomly shuffled.
    
    col_prime = randperm(col);
    training_inputs_prime = training_inputs(:,col_prime);
    training_results_prime = training_results(:,col_prime);
    
    %%
    % To speed up the computation, the data is divided into smaller data
    % set called minibatch.
    
    mini_batches = [];
    for j = 1:mini_batch_size:col
         mini_batches{end+1} = {training_inputs_prime(:,j:j+min(mini_batch_size-1,col-j)), training_results_prime(:,j:j+min(mini_batch_size-1,col-j))};
    end
    
    %%
    % Then, the optimal weights and biases are computed in each mini
    % batches. In each iteration, the value of the weights and biases are
    % updated and redirected to the next iteration.
    
    for j = 1:length(mini_batches)
        [W,b] = updateWeightBias(mini_batches{j}{1}, mini_batches{j}{2},eta,W,b,n,l);
    end
    
    %%
    % Here, The network performance is verified by using the optimal
    % weights and biases to perform digit recognition on the test data.
    corr_val(i) = validateNetwork(test_inputs, test_results, W, b, l);
    disp(['Epoch {',num2str(i),'} out of ',num2str(epoch),': ', num2str(corr_val(i)),'/',num2str(length(test_results))]);
    
end

%% 
% Once the network is well-trained with sufficiently large training data
% sets, we can store the resulting weights and biases and use them to
% perform the digit recognition process.