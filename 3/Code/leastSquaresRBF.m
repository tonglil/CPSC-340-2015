function [model] = leastSquaresRBF(X,y,sigma,lambda)
Xrbf = rbfBasis(X,X,sigma);

[n,~] = size(Xrbf);

% Solve least squares problem
w = (Xrbf'*Xrbf + lambda * eye(n))\Xrbf'*y;

model.X = X;
model.w = w;
model.sigma = sigma;
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
Xrbf = rbfBasis(Xtest,model.X,model.sigma);
yhat = Xrbf*model.w;
end

function [Xrbf] = rbfBasis(X1,X2,sigma)
N1 = size(X1,1);
N2 = size(X2,1);
D = size(X1,2);
Z = 1/sqrt(2*pi*sigma^2);
D = X1.^2*ones(D,N2) + ones(N1,D)*(X2').^2 - 2*X1*X2';
Xrbf = Z*exp(-D/(2*sigma^2));
end