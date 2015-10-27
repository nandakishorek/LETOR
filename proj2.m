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
M1 = 26;
M2 = 81;

% calculate sigma for both the datasets
[Sigma1, mu1] = calculateSigmaMu(trainingX1, M1);
[Sigma2, mu2] = calculateSigmaMu(trainingX2, M2);

% calculate phi for both the datasets
Phi1 = calculatePhi(trainingX1, M1, Sigma1, mu1);
Phi2 = calculatePhi(trainingX2, M2, Sigma2, mu2);

% regularization coefficients
lambda1 = 0;
lambda2 = 0;

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w1 = pinv((lambda1 * eye(M1)) + Phi1' * Phi1) * Phi1' * trainingT1;
w2 = pinv((lambda2 * eye(M2)) + Phi2' * Phi2) * Phi2' * trainingT2;

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
% initial weights M X 1
w01 = zeros(M1,1);

% number of iterations for gradient descent - E
numOfIters1 = 1000;

% learning rate 1 X E
eta1 = 1 * ones(1, numOfIters1);

% gradients M X E
dw1 = zeros(M1, numOfIters1);

fprintf('Performing stochastic gradient descent ...\n');
for i = 1 : numOfIters1
    for j = 1 : size(trainingX1, 1)
        dw1(:,i) = eta1(1,i) * ((trainingT1(j,1) - w01' * Phi1(j,:)') * Phi1(j,:)' + lambda1 * w01);
        w01 = w01 + dw1(:,i);
    end
end


% SGD
% initial weights M X 1
w02 = zeros(M2,1);

% number of iterations for gradient descent - E
numOfIters = 1000;

% learning rate 1 X E
eta2 = 1 * ones(1, numOfIters);

% gradients M X E
dw2 = ones(M2, numOfIters);

fprintf('Performing stochastic gradient descent ...\n');
for i = 1 : numOfIters
    for j = 1 : size(trainingX2, 1)
        %         dw2(:,i) = (trainingT2(j,1) - phi2(j,:) * w02) * phi2(j,:)';
        %         w02 = w02 + eta2(1,i) * dw2(:,i);
        dw2(:,i) = eta2(1,i) * ((trainingT2(j,1) - w02' * Phi2(j,:)') * Phi2(j,:)' + lambda2 * w02);
        w02 = w02 + dw2(:,i);
    end
end

save('proj2.mat');
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

function [Sigma, mu] = calculateSigmaMu(X, M)

rng default %

n = size(X,1);
d = size(X,2);

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M);
[idx, C] = kmeans(X, M);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu = C';

% mu = datasample(X, M);

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M);

cluster_variance = zeros(M,d);
for i = 1 : M
    temp = [];
    for j = 1 : length(idx)
        if j == i
            temp = [temp; X(j,:)];
        end
    end
    cluster_variance(i,:) = var(temp);
end

% the sigmaj for the basis functions
Sigma = zeros(d,d,M);
for j = 2 : M
    for i = 1 : n
        Sigma(:,:,j) = diag(cluster_variance(j,:));
    end
end
end