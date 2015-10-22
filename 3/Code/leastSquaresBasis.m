function [model] = leastSquaresBasis(X,y,degree)

X = polyBasis(X,degree);

% Solve least squares problem
w = (X' * X) \ X' * y;

model.degree = degree;
model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)

Xtest = polyBasis(Xtest,model.degree);

w = model.w;
yhat = Xtest * w;

end

function [Xnew] = polyBasis(X,degree)

[N,~] = size(X);

% Initialize a matrix with 0s with:
% Height = N
% Width = degree + 1
Xnew = zeros(N,degree+1);

% Calculate the polynomial order for each degree from 0
for j = 0:degree
    Xnew(:,j+1) = X.^j;
end

end