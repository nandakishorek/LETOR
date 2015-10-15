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

% number of training samples
n2 = size(trainingX2, 1);

% no. of dimensions of the training set
d2 = size(trainingX2, 2);

% total number of iterations
total = 200;
erms2 = zeros(1,total);

for M2 = 1 : total
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
    
    % determine design matrix N X M
    fprintf('Calculating the design matrix phi of size %d X %d ...\n', n2, M2);
    phi2 = ones(n2, M2);
    for j = 2 : M2
        siginv2 = pinv(cluster_variance(j)' * eye(d2));
        for i = 1 : n2
            temp = trainingX2(i,:)' - mu2(j);
            phi2(i,j) = exp(-1 * (temp' * siginv2 * temp) / 2);
        end
    end
    
    % closed form solution for the weights
    fprintf('Finding the closed form solution ...\n');
    w2 = pinv(phi2' * phi2) * phi2' * trainingT2;
    
    % sum of squares error
    error2 = sum((trainingT2 - (phi2 * w2)) .^ 2) / 2;
    
    % root mean square error
    erms2(1, M2) = sqrt(2 * error2 / n2);
    
end

% plot M vs ERMS
figure(2)
xaxis = linspace(0, total - 1, total);
plot(xaxis, erms2, 'r');
xlabel('M', 'Color','r');
ylabel('ERMS', 'Color', 'r');
