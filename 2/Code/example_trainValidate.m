clear all
load newsgroups.mat

[N,D] = size(X);
T = length(yvalidate);

% Compute validation error with decision tree
depth = 20;
model = decisionTree_InfoGain(X,y,depth);
yhat = model.predict(model,Xvalidate);
errorValidate = sum(yhat ~= yvalidate)/T;
fprintf('Validation error with decision tree: %.2f\n',errorValidate);

% Compute validation error with k-nearest neighbour
K = 5;
model = knn(X,y,K);
yhat = model.predict(model,Xvalidate);
errorValidate = sum(yhat ~= yvalidate)/T;
fprintf('Validation error with k-nearest neighbour: %.2f\n',errorValidate);

% Compute validation error with naive Bayes
model = naiveBayes(X,y);
yhat = model.predict(model,Xvalidate);
errorValidate = sum(yhat ~= yvalidate)/T;
fprintf('Validation error with naive Bayes: %.2f\n',errorValidate);

wordlistIndexes = model.wordlistIndexes;
for i = 1:max(wordlistIndexes)
    wordlist(wordlistIndexes(i,2:end))
end