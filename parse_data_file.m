% queryLevelNorm = importfile('MQ2007/Querylevelnorm.txt', 1, 69623);
% size(queryLevelNorm);
% save('querylevelnorm.mat');
load('querylevelnorm.mat');
X = Querylevelnorm(:, 2:end);
y = Querylevelnorm(:,1);