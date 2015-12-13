function [model] = softmaxClassifier(X,y)
% Classification using softmax loss

% n = samples: 500
% d = features: 3
% k = classes: 5

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k);     % Each column is a classifier

maxFunEvals = 500;  % Maximum number of evaluations of objective
verbose = 1;        % Whether or not to display progress of algorithm

% This is how you compute the function and gradient:
[f,g] = softMax(W(:),X,y,k);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(W(:),@softMax,X,y,k);

fprintf('f : %d\n', f);
fprintf('f2: %d\n', f2);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

W(:) = findMin(@softMax,W(:),maxFunEvals,verbose,X,y,k);

model.W = W;
model.predict = @predict;
end

function [f,g] = softMax(w,X,y,k)
[n,d] = size(X);
k = max(y);

% Reshape w's dimensions to "d x k"
% d = features: 3
% k = classes: 5
W = reshape(w, [d k]);

% Compute loss
f = 0;
for i = 1:n
    yi = y(i);      % the k
    Xi = X(i,:);    % 1 x 3

    %       (3 x 1)' (1 x 3)'   sum((1 x 3) (3 x 5))
    f = f - W(:,yi)' * Xi' + log(sum(exp(Xi * W)));
end

% Compute gradient
g = zeros(d,k);
for c = 1:k
    eachK = zeros(n,d);
    for i = 1:n
        Xi = X(i,:);
        
        numerator = exp(W(:,c)' * Xi') * Xi;
        denominator = sum(exp(Xi * W));

        eachK(i,:) = - Xi * (y(i) == c) + numerator / denominator;
    end
    g(:,c) = sum(eachK);
end

% Reshape the gradient's dimensions to a 1-D vector "1 x (d * k)"
g = g(:);
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end