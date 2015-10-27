function [] = proj2_real()
%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797
rng default %
clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% import data
load('querylevelnorm.mat');

% training set - 80% of dataset
trainingX = Querylevelnorm(1:55700, 2:end);
trainingT = Querylevelnorm(1:55700, 1);
trainInd1 = 1:55700;
trainInd1 = trainInd1';

% validation set - 10 % dataset
validationX = Querylevelnorm(55701:62661, 2:end);
validationT = Querylevelnorm(55701:62661, 1);
validInd1 = 55701:62661;
validInd1 = validInd1';

% testing set - 10% of dataset
testingX = Querylevelnorm(62662:end, 2:end);
testingT = Querylevelnorm(62662:end, 1);
testInd1 = 62662:size(Querylevelnorm,1);
testInd1 = testInd1';

% number of training samples
n1 = size(trainingX, 1);

% no. of dimensions of the training set
d1 = size(trainingX, 2);

% histograms for the dataset
% figure(1)
% histogram(trainingX(:,1));
% figure(2)
% histogram(trainingX(:,2));

% model complexity
M1 = 55;

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M1 );
[idx, C] = kmeans(trainingX, M1);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu1 = C';

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M1);
cluster_variance = zeros(M1,d1);
for i = 1 : M1
    temp = [];
    for j = 1 : length(idx)
        if j == i
            temp = [temp; trainingX(j,:)];
        end
    end
    cluster_variance(i,:) = var(temp);
end

% the sigmaj for the basis functions
Sigma1 = zeros(d1,d1,M1);
for j = 2 : M1
    for i = 1 : n1
        Sigma1(:,:,j) = diag(cluster_variance(j,:));
    end
end

% determine design matrix N X M
Phi1 = calculatePhi(trainingX, M1, Sigma1, mu1);

% regularization coefficient
lambda1 = 0;

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w1 = pinv((lambda1 * eye(M1)) + Phi1' * Phi1) * Phi1' * trainingT;

% sum of squares error and erms for the training set
[errorTest1, trainPer1] = calculateError(Phi1, trainingT, w1, size(trainingX, 1), lambda1)

% validation set
phiValid = calculatePhi(validationX, M1, Sigma1, mu1);
[errorVal1, validPer1] = calculateError(phiValid, validationT, w1, size(validationX, 1), 0)

% testing set
phiTest = calculatePhi(testingX, M1, Sigma1, mu1);
[errorTest1, testPer2] = calculateError(phiTest, testingT, w1, size(testingX, 1), 0);

figure(3)
y2 = Phi1 * w1;
xaxis = linspace(0, length(y2), length(y2));
plot(xaxis, trainingT, 'g', xaxis, y2, 'r');


% SGD
% initial weights M X 1
w01 = zeros(M1,1);

% number of iterations for gradient descent - E
numOfIters1 = 100;

% learning rate 1 X E
eta1 = 1 * ones(1, numOfIters1);

% gradients M X E
dw1 = zeros(M1, numOfIters1);

fprintf('Performing stochastic gradient descent ...\n');
for i = 1 : numOfIters1
    for j = 1 : n1
        dw1(:,i) = eta1(1,i) * ((trainingT(j,1) - w01' * Phi1(j,:)') * Phi1(j,:)' + lambda1 * w01);
        w01 = w01 + dw1(:,i);
    end
end

figure(4)
y2 = Phi1 * w01;
xaxis = linspace(0, length(y2), length(y2));
plot(xaxis, trainingT, 'g', xaxis, y2, 'r');

save('proj2_real.mat');
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