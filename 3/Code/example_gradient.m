
% Load data
load outliersData.mat

% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on

% Fit least-squares estimator
% model = leastSquaresGradient(X,y);

% Fit robust regression estimator
model = robustRegressionGradient(X,y,0.1);

% Draw model prediction
Xsample = [min(X):.01:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');
