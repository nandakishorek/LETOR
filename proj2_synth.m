function [] = proj2_synth()
%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% import data
load('synthetic.mat');

% transpose the input matrix to get 2000 X 10
x = x';

% training set - 80% of dataset
trainingX2 = x(1:1600, :);
trainingT2 = t(1:1600, 1);

% validation set - 10 % dataset
validationX2 = x(1601:1800, :);
validationT2 = t(1601:1800, 1);

% testing set - 10% of dataset
testingX2 = x(1801:size(x,1), :);
testingT2 = t(1801:size(t,1), 1);

% number of training samples
n2 = size(trainingX2, 1);

% no. of dimensions of the training set
d2 = size(trainingX2, 2);

% model complexity
M2 = 20;

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M2);
[idx2, C2] = kmeans(trainingX2, M2);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu2 = C2;

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M2);

cluster_variance = [];
for i = 1 : M2
    temp = [];
    for j = 1 : length(idx2)
        if j == i
            temp = [temp; trainingX2(j,:)];
        end
    end
    cluster_variance = [cluster_variance; var(temp)];
end

% the sigmaj for the basis functions
Sigma2 = zeros(d2,d2,M2);
for j = 2 : M2
    for i = 1 : n2
        Sigma2(:,:,j) = cluster_variance(j)' * eye(d2);
    end
end


phi2 = calculatePhi(trainingX2, M2, Sigma2, mu2);

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w2 = pinv(phi2' * phi2) * phi2' * trainingT2;

% sum of squares error for the training set
error2 = sum((trainingT2 - (phi2 * w2)) .^ 2) / 2;

% root mean square error for the training set
trainPer2 = sqrt(2 * error2 / n2)

% validation set design matrix
phiValid = calculatePhi(validationX2, M2, Sigma2, mu2);

% sum of squares error for the validation set
errorVal = sum((validationT2 - (phiValid * w2)) .^ 2) / 2;

% root mean square error for the validation set
validPer2 = sqrt(2 * errorVal / size(validationX2, 1))


% testing set design matrix
phiTest = calculatePhi(testingX2, M2, Sigma2, mu2);

% sum of squares error and erms for testing set
[errorTest, testPer2] = calculateError(phiTest, testingT2, w2, size(testingX2, 1))


% figure(2)
% y2 = phi2 * w2;
% xaxis = linspace(0, length(y2), length(y2));
% plot(xaxis, trainingT2, 'g', xaxis, y2, 'r');

save('proj2_synth');
end


function Phi = calculatePhi(X, M, Sigma, mu)

% number of training samples
n = size(X, 1);

% determine design matrix N X M
fprintf('Calculating the design matrix phi of size %d X %d ...\n', n, M);
Phi = ones(n, M);
for j = 2 : M
    for i = 1 : n
        temp = X(i,:)' - mu(j);
        siginv = pinv(Sigma(:,:,j));
        Phi(i,j) = exp(-1 * (temp' * siginv * temp) / 2);
    end
end

end

function [err, erms] = calculateError(phi, t, w, n)

% sum of squares error
err = sum((t - (phi * w)) .^ 2) / 2;

% root mean square error
erms = sqrt(2 * err / n);

end