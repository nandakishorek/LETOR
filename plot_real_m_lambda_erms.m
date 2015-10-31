clear; close all; clc
load('real_find_m.mat');
data = zeros(24 * length(lambda1),3);
data1 = zeros(24 * length(lambda1),3);
k = length(lambda1);
for i = 2 : 25
    for j =  1 : k
        index = k * (i-2) + j;
        data(index,1) = i;
        data(index,2) = lambda1(j,1);
        data(index,3) = ermsTraining1(i,j);
        
        data1(index,1) = i;
        data1(index,2) = lambda1(j,1);
        data1(index,3) = ermsValidation1(i,j);
    end
end
figure(100)
plot3(data(:,1), data(:,2), data(:,3),'b', data1(:,1), data1(:,2),data1(:,3), 'r');
legend('training','validation')
xlabel('M', 'Color','r');
ylabel('lambda', 'Color', 'r');
zlabel('erms', 'Color', 'r');