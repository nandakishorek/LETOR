function [] = synth_learning_curve()
%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% import data
load('querylevelnorm.mat');

% training set - 80% of dataset
trainingX1 = Querylevelnorm(1:55700, 2:end);
trainingT1 = Querylevelnorm(1:55700, 1);

% validation set - 10 % dataset
validationX1 = Querylevelnorm(55701:62661, 2:end);
validationT1 = Querylevelnorm(55701:62661, 1);

% testing set - 10% of dataset
testingX1 = Querylevelnorm(62662:end, 2:end);
testingT1 = Querylevelnorm(62662:end, 1);

% number of training samples
n1 = size(trainingX1, 1);

% no. of dimensions of the training set
d1 = size(trainingX1, 2);

% model complexity
M1 = 18;

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M1-1 );
rng default %
[idx, C] = kmeans(trainingX1, M1-1, 'MaxIter',1000);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu1 = C';
mu1 = [zeros(d1,1) mu1];

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M1);

% the sigmaj for the basis functions
Sigma1 = zeros(d1,d1,M1);
for j = 1 : M1
    Sigma1(:,:,j) = 0.1 * eye(d1);
end

% determine design matrix N X M
Phi1 = calculatePhi(trainingX1, M1, Sigma1, mu1);


% regularization coefficient
lambda1 = 0.01;

total = size(validationX1,1);
ermsTraining = zeros(1,total);
ermsValidation = zeros(1,total);

phiValid = calculatePhi(validationX1, M1, Sigma1, mu1);

for i = 1 : total
    
    % closed form solution for the weights
    fprintf('Finding the closed form solution ...\n');
    w1 = pinv((lambda1 * eye(M1)) + Phi1(1:i,:)' * Phi1(1:i,:)) * Phi1(1:i,:)' * trainingT1(1:i,:);
    
    % sum of squares error and erms for the training set
    [errorTrain2, ermsTraining(1,i)] = calculateError(Phi1(1:i,:), trainingT1(1:i,:), w1, i, 0);
    
    % validation set
    [errorVal2, ermsValidation(1,i)] = calculateError(phiValid, validationT1, w1, size(phiValid, 1), 0);
end

figure(3)
plot(1:total, ermsTraining, 'b', 1:total, ermsValidation, 'r');
legend('training','validation')
xlabel('N', 'Color','r');
ylabel('ERMS', 'Color', 'r');

save('real_learning_curve.mat');
end


function Phi = calculatePhi(X, M, Sigma, mu)

% number of training samples
n = size(X, 1);

% determine design matrix N X M
fprintf('Calculating the design matrix phi of size %d X %d ...\n', n, M);
Phi = ones(n, M);
for j = 2 : M
    siginv = inv(Sigma(:,:,j));
    for i = 1 : n
        temp = X(i,:)' - mu(:,j);
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