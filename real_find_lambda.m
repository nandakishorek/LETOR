%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797

function [] = real_find_lambda()
clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% import data
load('querylevelnorm.mat');

% training set - 80% of dataset
trainingX = Querylevelnorm(1:55700, 2:end);
trainingT = Querylevelnorm(1:55700, 1);

% validation set - 10 % dataset
validationX = Querylevelnorm(55701:62661, 2:end);
validationT = Querylevelnorm(55701:62661, 1);

% testing set - 10% of dataset
testingX = Querylevelnorm(62662:end, 2:end);
testingT = Querylevelnorm(62662:end, 1);

% number of training samples
n1 = size(trainingX, 1);

% no. of dimensions of the training set
d1 = size(trainingX, 2);

% regularization coefficient
lambda1 = linspace(0,1,11)';

% total number of iterations
total = length(lambda1);
ermsTraining = zeros(1,total);
ermsValidation = zeros(1,total);
ermsTest = zeros(1,total);

M1 = 26;

rng default %
% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M1);
[idx1, C1] = kmeans(trainingX, M1);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu1 = C1';

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M1);

cluster_variance = zeros(M1,d1);
for i = 1 : M1
    temp = [];
    for j = 1 : length(idx1)
        if j == i
            temp = [temp; trainingX(j,:)];
        end
    end
    cluster_variance(i,:) = var(temp);
    %     cluster_variance(i,:) = 1 * ones(1, d2);
    %     cluster_variance(i,1) = 0.5;
end

% the sigmaj for the basis functions
Sigma1 = zeros(d1,d1,M1);
for j = 2 : M1
    for i = 1 : n1
        Sigma1(:,:,j) = diag(cluster_variance(j,:));
    end
end


Phi1 = calculatePhi(trainingX, M1, Sigma1, mu1);
for k = 1 : total
    % closed form solution for the weights
    fprintf('Finding the closed form solution ...\n');
    w1 = pinv((lambda1(k,1) * eye(M1)) + Phi1' * Phi1) * Phi1' * trainingT;
    
    % sum of squares error and erms for the training set
    [errorTrain1, ermsTraining(1,k)] = calculateError(Phi1, trainingT, w1, size(trainingX, 1), lambda1(k,1));
    
    % validation set
    phiValid = calculatePhi(validationX, M1, Sigma1, mu1);
    [errorVal1, ermsValidation(1,k)] = calculateError(phiValid, validationT, w1, size(validationX, 1), 0);
    
    % testing set
    phiTest = calculatePhi(testingX, M1, Sigma1, mu1);
    [errorTest1, ermsTest(1,k)] = calculateError(phiTest, testingT, w1, size(testingX, 1), 0);
    
end

% plot M vs ERMS
figure(2)
plot(lambda1, ermsTraining, 'b', lambda1, ermsValidation, 'r', lambda1, ermsTest, 'g');
legend('training','validation','testing');
xlabel('lambda', 'Color','r');
ylabel('ERMS', 'Color', 'r');

save('real_find_lambda.mat');
end

function Phi = calculatePhi(X, M, Sigma, mu)

% number of training samples
n = size(X, 1);

% determine design matrix N X M
fprintf('Calculating the design matrix phi of size %d X %d ...\n', n, M);
Phi = ones(n, M);
for j = 2 : M
    for i = 1 : n
        temp = X(i,:)' - mu(:,j);
        siginv = inv(Sigma(:,:,j));
        Phi(i,j) = exp(-1 * (temp' * siginv * temp) / 2);
    end
end

end

function [err, erms] = calculateError(phi, t, w, n, lambda)

% sum of squares error
err = sum((t - (phi * w)) .^ 2) / 2 + (lambda * (w' * w) / 2);

% root mean square error
erms = sqrt(2 * err / n);

end
