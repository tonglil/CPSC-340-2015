function [model] = weightedLeastSquares(X,y,Z)

% Use derived weighted least squares function
w = (X'*Z*X)\X'*Z*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end