function [model] = logLinearClassifier(X,y)
% Classification using logistic loss

% n = samples
% d = features
% k = classes

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k);     % Each column is a classifier

maxFunEvals = 400;  % Maximum number of evaluations of objective
verbose = 1;        % Whether or not to display progress of algorithm

for c = 1:k
    yc = ones(n,1);     % Treat class 'c' as (+1)
    yc(y ~= c) = -1;    % Treat other classes as (-1)
    W(:,c) = findMin(@logisticLoss,W(:,1),maxFunEvals,verbose,X,yc);
end

model.W = W;
model.predict = @predict;
end

function [f,g] = logisticLoss(w,X,y)
yXw = y.*(X*w);
f = sum(log(1 + exp(-yXw)));    % Function value
g = -X'*(y./(1+exp(yXw)));      % Gradient
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end