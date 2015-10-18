clear all
load DTdata.mat

[N,D] = size(X);
T = length(ytest);

Xfold1 = X(1:end/2,:);      % select rows 1 to 2500, all columns
Xfold2 = X((end/2+1):end,:);
y1 = y(1:end/2,:);
y2 = y((end/2+1):end,:);

maxDepth = 15;
xAxis = [];
yAxis = [];

for i = 1:maxDepth
    errorAverage = folds(Xfold1,Xfold2,y1,y2,i);
%     model = decisionTree_InfoGain(X,y,i);
%     
%     yhat = model.predictFunc(model,X);
%     errorTrain = sum(yhat ~= y)/N;
%     
%     yhat = model.predictFunc(model,Xtest);
%     errorTest = sum(yhat ~= ytest)/T
    
    xAxis = [xAxis, i];
    yAxis = [yAxis, errorAverage];
end

scatter(xAxis, yAxis);

disp(yAxis);
min(yAxis)