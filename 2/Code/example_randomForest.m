clear all
load iris.mat

[N,D] = size(X);
T = length(yvalidate);

% Compute validation error with decision tree
depth = 4;
model = decisionTree_InfoGain(X,y,depth);
yhat = model.predict(model,Xvalidate);
errorValidate = sum(yhat ~= yvalidate)/T;
% fprintf('Validation error with decision tree: %.6f\n',errorValidate);

% Compute validation error with bootstrapped decision tree
nTrees = 100;
model = randomForest(X,y,depth,nTrees);
yhat = model.predict(model,Xvalidate);
errorValidate = sum(yhat ~= yvalidate)/T;
fprintf('Validation error with random forest: %.6f\n',errorValidate);