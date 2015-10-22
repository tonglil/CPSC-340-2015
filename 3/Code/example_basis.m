
% Load data
load basisData.mat

% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on;

% Fit least-squares estimator
% model = leastSquares(X,y);

% Fit least-squares with bias estimator
model = simpleLeastSquares(X,y);

% Draw model prediction
Xsample = [min(X):.1:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');