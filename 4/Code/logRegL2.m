function [model] = logRegL2(X,y,lambda)
[~,d] = size(X);

maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm
w0 = zeros(d,1);

model.w = findMin(@logisticLoss,w0,maxFunEvals,verbose,X,y,lambda);
model.predict = @(model,X)sign(X*model.w); % Predictions by taking sign
end

function [f,g] = logisticLoss(w,X,y,lambda)
yXw = y .* (X * w);
f = sum(log(1 + exp(-yXw))) + lambda / 2 * norm(w) ^ 2; % Function value
g = -X' * (y ./ (exp(yXw) + 1)) + lambda * w; % Gradient
end