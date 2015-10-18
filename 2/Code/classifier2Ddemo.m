% Load dataset
load binary.mat

[N,D] = size(X);

% Fit decision tree
% model = decisionTree_InfoGain(X,y,9);

K = 50;
trainingErrors = zeros(K,1);

% Fit K-nearest neighbors
for i = 1:K
    model = knn(X,y,i);

    % Compute training error
    yhat = model.predict(model,X);
    trainingErrors(i) = sum(yhat ~= y)/N;
end

plot(1:K, trainingErrors);
xlabel('k');
ylabel('Training Errors');

% Show data and decision boundaries
% classifier2Dplot(X,y,model);