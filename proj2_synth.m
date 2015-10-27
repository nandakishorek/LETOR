function [] = proj2_synth()
%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797
rng default %
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

% % histograms for the dataset
% figure(1)
% histogram(trainingX2(:,1));
% figure(2)
% histogram(trainingX2(:,2));

% model complexity
M2 = 60;

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M2);
[idx2, C2] = kmeans(trainingX2, M2);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu2 = C2';

% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M2);

cluster_variance = zeros(M2,d2);
for i = 1 : M2
    temp = [];
    for j = 1 : length(idx2)
        if j == i
            temp = [temp; trainingX2(j,:)];
        end
    end
    cluster_variance(i,:) = var(temp);
%     cluster_variance(i,:) = 0.5 * ones(1, d2);
%     cluster_variance(i,1) = 0.5;
end

% the sigmaj for the basis functions
Sigma2 = zeros(d2,d2,M2);
for j = 2 : M2
    for i = 1 : n2
        Sigma2(:,:,j) = diag(cluster_variance(j,:));
    end
end


phi2 = calculatePhi(trainingX2, M2, Sigma2, mu2);

% regularization coefficient
lambda2 = 0;

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w2 = pinv((lambda2 * eye(M2)) + phi2' * phi2) * phi2' * trainingT2;

% sum of squares error and erms for the training set
[errorTrain2, trainPer2] = calculateError(phi2, trainingT2, w2, size(trainingX2, 1), lambda2)

% validation set
phiValid = calculatePhi(validationX2, M2, Sigma2, mu2);
[errorVal2, validPer2] = calculateError(phiValid, validationT2, w2, size(validationX2, 1), 0)

% testing set
phiTest = calculatePhi(testingX2, M2, Sigma2, mu2);
[errorTest2, testPer2] = calculateError(phiTest, testingT2, w2, size(testingX2, 1), 0);


% SGD
% initial weights M X 1
w02 = zeros(M2,1);

% number of iterations for gradient descent - E
numOfIters = 500;

% learning rate 1 X E
eta2 = 1 * ones(1, numOfIters);

% gradients M X E
dw2 = ones(M2, numOfIters);

fprintf('Performing stochastic gradient descent ...\n');
for i = 1 : numOfIters
    for j = 1 : n2
%         dw2(:,i) = (trainingT2(j,1) - phi2(j,:) * w02) * phi2(j,:)';
%         w02 = w02 + eta2(1,i) * dw2(:,i);
        dw2(:,i) = eta2(1,i) * ((trainingT2(j,1) - w02' * phi2(j,:)') * phi2(j,:)' + lambda2 * w02);
        w02 = w02 + dw2(:,i);
    end
end

figure(3)
y2 = phi2 * w2;
xaxis = linspace(0, length(y2), length(y2));
plot(xaxis, trainingT2, 'g', xaxis, y2, 'r');

figure(4)
y3 = phi2 * w02;
xaxis = linspace(0, length(y3), length(y3));
plot(xaxis, trainingT2, 'g', xaxis, y3, 'r');

save('proj2_synth.mat');
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