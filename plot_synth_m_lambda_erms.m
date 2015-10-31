clear; close all; clc
load('synth_find_m.mat');
data = zeros(99 * length(lambda2),3);
data1 = zeros(99 * length(lambda2),3);
k = length(lambda2);
for i = 2 : 99
    for j =  1 : k
        index = k * (i-2) + j;
        data(index,1) = i;
        data(index,2) = lambda2(j,1);
        data(index,3) = ermsTraining(i,j);
        
        data1(index,1) = i;
        data1(index,2) = lambda2(j,1);
        data1(index,3) = ermsValidation(i,j);
    end
end
figure(100)
plot3(data(:,1), data(:,2), data(:,3),'b', data1(:,1), data1(:,2),data1(:,3), 'r');
legend('training','validation')
xlabel('M', 'Color','r');
ylabel('lambda', 'Color', 'r');
zlabel('erms', 'Color', 'r');