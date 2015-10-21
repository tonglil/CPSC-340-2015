%% Load Data
warning off all;
close all;
clear all;
load nonLinearData.mat;
[n,d] = size(X);

%% Plotting Code
plot(X,y,'b.');
hold on;
plot(Xtest,ytest,'g.');
xl = xlim;
yl = ylim;
Xvals = [xl(1):.1:xl(2)]';
pause(.1);

%% Train

% Split the data into training and test sets
split = 50;

xTraining = X(1:split,:);
yTraining = y(1:split,:);
xValidation = X(split+1:end,:);
yValidation = y(split+1:end,:);

errorLowest = 100;

% Display result of fitting with RBF kernel
for sigma = 2.^[3:-1:-4]
    for lambda = 2.^[-12:2]
        % Train on xTraining, validate on xValidation
        model = leastSquaresRBF(xTraining,yTraining,sigma,lambda);
        yHat = model.predict(model,xValidation);
        error = mean(abs(yHat-yValidation));
        
        % Determine if error is lower
        if error < errorLowest
            errorLowest = error;
            sigmaLowest = sigma;
            lambdaLowest = lambda;
        end
        
        fprintf('Test error sigma = %f, lambda = %f, is %f\n',sigma,lambda,error);
    end
end

%% Test

fprintf('Best sigma = %f\n',sigmaLowest);
fprintf('Best lambda = %f\n',lambdaLowest);

% Train on full X
model = leastSquaresRBF(X,y,sigmaLowest,lambdaLowest);
% Test on Xtest
yHat = model.predict(model,Xtest);
% Get the final error
error = mean(abs(yHat-ytest));

fprintf('Test error using best is %f\n',error);

%% Plot Results
figure(1);
clf;
plot(X,y,'b.');
hold on;
plot(Xtest,ytest,'g.');
yvals = model.predict(model,Xvals);
plot(Xvals,yvals,'r-');
legend({'Train','Test'});
ylim(yl);
title(sprintf('RBF Basis (sigma = %f, lambda = %f)',sigmaLowest,lambdaLowest));