function [model] = simpleLeastSquares(X,y)

[N,~] = size(X);

% Make a matrix with 1s and X
newX = ones(N,2);
newX(:,2) = X;

X = newX;

% Solve least squares problem
w = (X' * X) \ X' * y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)

[N,~] = size(Xtest);

% Make a matrix with 1s and X
newXtest = ones(N,2);
newXtest(:,2) = Xtest;

Xtest = newXtest;

w = model.w;
yhat = Xtest * w;

end