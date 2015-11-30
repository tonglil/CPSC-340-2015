clear all
close all
load multiData.mat

% Data is already roughly standardized, but let's add bias
[n,d] = size(X);
X = [ones(n,1) X];

% Fit least-squares classifier
model = leastSquaresClassifier(X,y);

% Compute validation error
t = size(Xvalidate,1);
Xvalidate = [ones(t,1) Xvalidate];
yhat = model.predict(model,Xvalidate);
errors = sum(yvalidate ~= yhat)/t

% Plot result
k = max(y);
classifier2Dplot(X,y,k,model);