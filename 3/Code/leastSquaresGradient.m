function [model] = leastSquaresGradient(X,y)

[n,d] = size(X);

% Initial guess
w0 = zeros(d,1);

% This is how you compute the function and gradient:
[f,g] = funObj(w0,X,y);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w0,@funObj,X,y);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

% Solve least squares problem
w = findMin(@funObj,w0,100,X,y);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end

function [f,g] = funObj(w,X,y)
    f = (1/2)*sum((X*w-y).^2);
    g = X'*(X*w - y);
end