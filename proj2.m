function [] = proj2()
%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% import real data
load('querylevelnorm.mat');

% training set - 80% of dataset
trainingX1 = Querylevelnorm(1:55700, 2:end);
trainingT1 = Querylevelnorm(1:55700, 1);
trainInd1 = 1:55700;
trainInd1 = trainInd1';

% validation set - 10 % dataset
validationX1 = Querylevelnorm(55701:62661, 2:end);
validationT1 = Querylevelnorm(55701:62661, 1);
validInd1 = 55701:62661;
validInd1 = validInd1';

% testing set - 10% of dataset
testingX1 = Querylevelnorm(62662:end, 2:end);
testingT1 = Querylevelnorm(62662:end, 1);
testInd1 = 62662:size(Querylevelnorm,1);
testInd1 = testInd1';

% import synthetic data
load('synthetic.mat');

% transpose the input matrix to get 2000 X 10
x = x';

% training set - 80% of dataset
trainingX2 = x(1:1600, :);
trainingT2 = t(1:1600, 1);
trainInd2 = 1:1600;
trainInd2 = trainInd2';

% validation set - 10 % dataset
validationX2 = x(1601:1800, :);
validationT2 = t(1601:1800, 1);
validInd2 = 1601:1800;
validInd2 = validInd2';

% testing set - 10% of dataset
testingX2 = x(1801:size(x,1), :);
testingT2 = t(1801:size(t,1), 1);
testInd2 = 1801:size(x,1);
testInd2 = testInd2';

% model complexity
M1 = 18;
M2 = 40;

% calculate sigma for both the datasets
[Sigma1, mu1] = calculateSigmaMuReal(trainingX1, M1);
[Sigma2, mu2] = calculateSigmaMuSynth(trainingX2, M2);

% calculate phi for both the datasets
Phi1 = calculatePhi(trainingX1, M1, Sigma1, mu1);
Phi2 = calculatePhi(trainingX2, M2, Sigma2, mu2);

% regularization coefficients
lambda1 = 0;
lambda2 = 0;

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w1 = ((lambda1 * eye(M1)) + Phi1' * Phi1) \ Phi1' * trainingT1;
w2 = ((lambda2 * eye(M2)) + Phi2' * Phi2) \ Phi2' * trainingT2;

% training error
[errorTrain1, trainPer1] = calculateError(Phi1, trainingT1, w1, size(trainingX1, 1), lambda1)
[errorTrain2, trainPer2] = calculateError(Phi2, trainingT2, w2, size(trainingX2, 1), lambda2)

% validation
phiValid1 = calculatePhi(validationX1, M1, Sigma1, mu1);
phiValid2 = calculatePhi(validationX2, M2, Sigma2, mu2);
[errorVal1, validPer1] = calculateError(phiValid1, validationT1, w1, size(validationX1, 1), 0)
[errorVal2, validPer2] = calculateError(phiValid2, validationT2, w2, size(validationX2, 1), 0)

% test
phiTest1 = calculatePhi(testingX1, M1, Sigma1, mu1);
phiTest2 = calculatePhi(testingX2, M2, Sigma2, mu2);
[errorTest1, testPer1] = calculateError(phiTest1, testingT1, w1, size(testingX1, 1), 0)
[errorTest2, testPer2] = calculateError(phiTest2, testingT2, w2, size(testingX2, 1), 0)


% figure(2)
% y2 = Phi2 * w2;
% xaxis = linspace(0, length(y2), length(y2));
% plot(xaxis, trainingT2, 'g', xaxis, y2, 'r');


% SGD
% number of iterations for gradient descent - E
numOfIters1 = 55700;

% initial weights M X 1
w01 = zeros(M1, 1);

% learning rate 1 X E
eta1 = 0.2 * ones(1, numOfIters1);

% gradients M X E
dw1 = zeros(M1, numOfIters1);

fprintf('Performing stochastic gradient descent ...\n');
tempW01 = w01;
for i = 1 : numOfIters1
    dw1(:,i) = eta1(1,i) * ((trainingT1(i,1) - tempW01' * Phi1(i,:)') * Phi1(i,:)' + lambda1 * tempW01);
    tempW01 = tempW01 + dw1(:,i);
end

norm(w1-tempW01)/norm(w1)

% SGD
% number of iterations for gradient descent - E
numOfIters2 = 1600;

% initial weights M X 1
w02 = zeros(M2,1);

% learning rate 1 X E
eta2 = 0.1 * ones(1, numOfIters2);

% gradients M X E
dw2 = ones(M2, numOfIters2);

fprintf('Performing stochastic gradient descent ...\n');
tempW02 = w02;
prevError = 1;
for i = 1 : numOfIters2
    dw2(:,i) = eta2(1,i) * ((trainingT2(i,1) - tempW02' * Phi2(i,:)') * Phi2(i,:)' + lambda2 * tempW02);
    tempW02 = tempW02 + dw2(:,i);
    [sgdErr, sgdErms] = calculateError(Phi2(i,:), trainingT2(i,1), tempW02, 1, lambda1);
    if (sgdErms - prevError) > 0.0001
        eta2(1,i+1) = eta2(1,i) / 2;
        tempW02 = tempW02 - dw2(:,i);
    end
    prevError = sgdErms;
end
norm(w2-tempW02) / norm(w2)

save('proj2.mat', 'M1', 'M2', 'mu1', 'mu2', 'Sigma1', 'Sigma2', 'lambda1', 'lambda2', 'trainInd1', 'trainInd2', 'validInd1', 'validInd2', 'w1', 'w2', 'w01', 'w02', 'dw1', 'dw2', 'eta1', 'eta2', 'trainPer1', 'trainPer2', 'validPer1', 'validPer2');
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
        Phi(i,j) = exp(-1 * (temp' / Sigma(:,:,j) * temp) / 2);
    end
end
end

function [err, erms] = calculateError(phi, t, w, n, lambda)
% sum of squares error
err = sum((t - (phi * w)) .^ 2) / 2 + (lambda * (w' * w) / 2);
% root mean square error
erms = sqrt(2 * err / n);
end

function [Sigma, mu] = calculateSigmaMuReal(X, M)

d = size(X,2);

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M-1);
rng default %
[idx, C] = kmeans(X, M-1, 'MaxIter',1000);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu = C';
mu = [zeros(d,1) mu];

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M);

% the sigmaj for the basis functions
Sigma = zeros(d,d,M);
for j = 1 : M
    Sigma(:,:,j) = 0.1 * eye(d);
end
end

function [Sigma, mu] = calculateSigmaMuSynth(X, M)

d = size(X,2);

% find the clusters for the datapoints
rng default %
fprintf('Finding %d clusters ...\n', M-1);
[idx, C] = kmeans(X, M-1);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu = C';
mu = [zeros(d,1) mu];

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M);

variance = var(X);
% the sigmaj for the basis functions
Sigma = zeros(d,d,M);
for j = 1 : M
    Sigma(:,:,j) = diag(variance);
end
end