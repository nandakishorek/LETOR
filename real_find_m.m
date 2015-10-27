%% Project 2 - Learning to Rank using Linear Regression
%% Nandakishore Krishna
%% Person number : 50169797
function [] = real_find_m()

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

% total number of iterations
total = 50;
ermsTraining1 = zeros(1,total);
ermsValidation1 = zeros(1,total);
ermsTest1 = zeros(1,total);

for M1 = 1 : total
    
    % reset random number generator so that the output of kmeans is deterministic
    rng default %
    
    % find the clusters for the datapoints
    fprintf('Finding %d clusters ...\n', M1 );
    [idx1, C1] = kmeans(trainingX1, M1);
    
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
                temp = [temp; trainingX1(j,:)];
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
    Phi1 = calculatePhi(trainingX1, M1, Sigma1, mu1);
    
    % regularization coefficient
    lambda1 = 0;
    
    % closed form solution for the weights
    fprintf('Finding the closed form solution ...\n');
    w1 = pinv((lambda1 * eye(M1)) + Phi1' * Phi1) * Phi1' * trainingT1;
    
    % sum of squares error and erms for the training set
    [errorTest1, ermsTraining1(1,M1)] = calculateError(Phi1, trainingT1, w1, size(trainingX1, 1), lambda1);
    
    % validation set
    phiValid1 = calculatePhi(validationX1, M1, Sigma1, mu1);
    [errorVal1, ermsValidation1(1,M1)] = calculateError(phiValid1, validationT1, w1, size(validationX1, 1), 0);
    
    % testing set
    phiTest1 = calculatePhi(testingX1, M1, Sigma1, mu1);
    [errorTest1, ermsTest1(1,M1)] = calculateError(phiTest1, testingT1, w1, size(testingX1, 1), 0);
    
end

% plot M vs ERMS
figure(10)
xaxis = linspace(0, total - 1, total);
plot(xaxis, ermsTraining1, 'b', xaxis, ermsValidation1, 'r');
legend('training','validation')
xlabel('M', 'Color','r');
ylabel('ERMS', 'Color', 'r');

save('real_find_m.mat');
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