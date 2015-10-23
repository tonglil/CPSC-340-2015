%% Load data
load outliersData.mat;

%% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on;

%% Fit least-squares estimator
% model = leastSquares(X,y);

%% Fit weighted least-squares estimator
[N,D] = size(X);
weights = zeros(N);

for x = 1:400
    weights(x,x) = 1;
end

for x = 401:500
    weights(x,x) = 0.1;
end

model = weightedLeastSquares(X,y,weights);

%% Draw model prediction
Xsample = [min(X):.01:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');