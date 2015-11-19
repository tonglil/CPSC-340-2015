function [model] = recommendGlobalMean(X,y)
beta = mean(y);

model.beta = beta;
model.predict = @predict;
end

function [y] = predict(model,X)
t = size(X,1);
beta = model.beta;

y = beta*ones(t,1);
end