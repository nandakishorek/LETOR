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
trainingX = Querylevelnorm(1:55700, 2:end);
trainingT = Querylevelnorm(1:55700, 1);

% validation set - 10 % dataset
validationX = Querylevelnorm(55700:62661, 2:end);
validationT = Querylevelnorm(55700:62661, 1);

% testing set - 10% of dataset
testingX = Querylevelnorm(62662:end, 2:end);
testingT = Querylevelnorm(62662:end, 1);

% number of training samples
n = size(trainingX, 1);

% no. of dimensions of the training set
d = size(trainingX, 2);

% histograms for the dataset
% figure(1)
% histogram(trainingX(:,1));
% figure(2)
% histogram(trainingX(:,2));

% calculate vaiance for the training set
variance = var(trainingX);

% model complexity
M1 = 10;

% % find the clusters for the datapoints
% fprintf('Finding %d clusters ...\n', M1 );
% [idx, C] = kmeans(trainingX, M1);
% 
% % centres for the basis functions D X M
% % we assign centroids of the clusters to muj
% mu1 = C;

% centres for the basis functions D X M
mu1 = ones(d, M1);

% pick five random points as centres, in this case samples numbered 11 to 11+M1
for i = 1 : M1
    mu1(:,i) = trainingX(i + 10,:)';
end


% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M1);
sig = zeros(d,d);
for i = 1 : d
    for j = 1 : d
        if i == j
            sig(i,j) = variance(i) / 10;
        end
    end
end
siginv = pinv(sig);
% siginv = pinv(cov(trainingX) * eye(d));

% determine design matrix N X M
fprintf('Calculating the design matrix phi of size %d X %d ...\n', n, M1);
phi = ones(n, M1); 
for j = 2 : M1
    for i = 1 : n
        temp = trainingX(i,:)' - mu1(j);
        phi(i,j) = exp(-1 * (temp' * siginv * temp) / 2);
    end
end

% regularization coefficient
lambda = 100;

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w1 = pinv((lambda * eye(M1)) + phi' * phi) * phi' * trainingT;

% sum of squares error
error1 = (sum((trainingT - (phi * w1)) .^ 2) / 2) + (lambda * w1' * w1 / 2)

y = phi * w1;
xaxis = linspace(0, length(y), length(y));
plot(xaxis, trainingT, 'g', xaxis, y, 'r');
