function [model] = logRegL1(X,y,lambda)
[~,d] = size(X);

maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm
w0 = zeros(d,1);

model.w = findMinL1(@logisticLoss,w0,lambda,maxFunEvals,verbose,X,y);
model.predict = @(model,X)sign(X*model.w); % Predictions by taking sign
end

function [f,g] = logisticLoss(w,X,y)
yXw = y .* (X * w);
f = sum(log(1 + exp(-yXw))); % Function value
g = -X' * (y ./ (exp(yXw) + 1)); % Gradient
end