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
M1 = 44;
M2 = 55;

% calculate sigma for both the datasets
[Sigma1, mu1] = calculateSigmaMu(trainingX1, M1);
[Sigma2, mu2] = calculateSigmaMu(trainingX2, M2);

mu1 = mu1';
mu2 = mu2';

% calculate phi for both the datasets
Phi1 = calculatePhi(trainingX1, M1, Sigma1, mu1');
Phi2 = calculatePhi(trainingX2, M2, Sigma2, mu2');

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
phiValid1 = calculatePhi(validationX1, M1, Sigma1, mu1');
phiValid2 = calculatePhi(validationX2, M2, Sigma2, mu2');
[errorVal1, validPer1] = calculateError(phiValid1, validationT1, w1, size(validationX1, 1), lambda1)
[errorVal2, validPer2] = calculateError(phiValid2, validationT2, w2, size(validationX2, 1), lambda2)

% test
phiTest1 = calculatePhi(testingX1, M1, Sigma1, mu1');
phiTest2 = calculatePhi(testingX2, M2, Sigma2, mu2');
[errorTest1, testPer1] = calculateError(phiTest1, testingT1, w1, size(testingX1, 1), lambda1)
[errorTest2, testPer2] = calculateError(phiTest2, testingT2, w2, size(testingX2, 1), lambda2)


% figure(2)
% y2 = Phi2 * w2;
% xaxis = linspace(0, length(y2), length(y2));
% plot(xaxis, trainingT2, 'g', xaxis, y2, 'r');

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
        temp = X(i,:)' - mu(j);
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

n = size(X,1);
d = size(X,2);

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M);
[idx, C] = kmeans(X, M);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu = C;

% mu = datasample(X, M);

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M);

cluster_variance = [];
for i = 1 : M
    temp = [];
    for j = 1 : length(idx)
        if j == i
            temp = [temp; X(j,:)];
        end
    end
    cluster_variance = [cluster_variance; var(temp)];
%     cluster_variance(i,1) = 1;
end

% the sigmaj for the basis functions
Sigma = zeros(d,d,M);
for j = 2 : M
    for i = 1 : n
        Sigma(:,:,j) = cluster_variance(j,:)' * eye(d);
    end
end
end