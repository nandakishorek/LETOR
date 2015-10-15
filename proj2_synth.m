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
testingX2 = x(1801:end, :);
testingT2 = t(1801:end, 1);

% number of training samples
n2 = size(trainingX2, 1);

% no. of dimensions of the training set
d2 = size(trainingX2, 2);

% histograms for the dataset
% figure(1)
% histogram(trainingX(:,1));
% figure(2)
% histogram(trainingX(:,2));

% calculate vaiance for the training set
variance2 = var(trainingX2);

% model complexity
M2 = 80;

% find the clusters for the datapoints
fprintf('Finding %d clusters ...\n', M2);
[idx2, C2] = kmeans(trainingX2, M2);

% centres for the basis functions D X M
% we assign centroids of the clusters to muj
mu2 = C2;

% % centres for the basis functions D X M
% mu2 = ones(d2, M2);
% 
% % pick five random points as centres, in this case samples numbered 11 to 11+M1
% for i = 1 : M2
%     mu2(:,i) = trainingX2(i + 10,:)';
% end


% spread for the Gaussian radial functions
fprintf('Calculating the spread for the %d Gaussian radial functions ...\n', M2);

% clusters = zeros(n2,d2,M2);
% indices = ones(M2,1);
% for i = 1 : n2
%     temp = clusters(:,:,idx2(i,1));
%     j = indices(idx2(i,1),1);
%     indices(idx2(i,1),1) = j + 1;
%     temp(j, :) = trainingX2(idx2(i,1),:);
% end
% [clusters{1:M2}] = deal([]);
% for i = 1 : n2
%     temp = clusters{idx2(i,1)};
%     temp = [temp; trainingX2(idx2(i,1),:)];
%     clusters{idx2(i,1)} = temp;
% end

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

% sig2 = zeros(d2,d2);
% for i = 1 : d2
%     for j = 1 : d2
%         if i == j
%             sig2(i,j) = variance2(i)/0.271;
%         end
%     end
% end
% siginv2 = pinv(sig2);
% siginv = pinv(cov(trainingX) * eye(d));

% determine design matrix N X M
fprintf('Calculating the design matrix phi of size %d X %d ...\n', n2, M2);
phi2 = ones(n2, M2);
for j = 2 : M2
    for i = 1 : n2
        temp = trainingX2(i,:)' - mu2(j);
%         siginv2 = pinv(cov(clusters{j}) * eye(d2));
        siginv2 = pinv(cluster_variance(j)' * eye(d2));
        phi2(i,j) = exp(-1 * (temp' * siginv2 * temp) / 2);
    end
end

% closed form solution for the weights
fprintf('Finding the closed form solution ...\n');
w2 = pinv(phi2' * phi2) * phi2' * trainingT2;

% sum of squares error
error2 = sum((trainingT2 - (phi2 * w2)) .^ 2) / 2

figure(2)
y2 = phi2 * w2;
xaxis = linspace(0, length(y2), length(y2));
plot(xaxis, trainingT2, 'g', xaxis, y2, 'r');