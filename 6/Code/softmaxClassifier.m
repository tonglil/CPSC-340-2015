function [model] = softmaxClassifier(X,y)
% Classification using softmax loss

% n = samples: 500
% d = features: 3
% k = classes: 5

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k);     % Each column is a classifier

maxFunEvals = 400;  % Maximum number of evaluations of objective
verbose = 1;        % Whether or not to display progress of algorithm

W(:) = findMin(@softMax,W(:),maxFunEvals,verbose,X,y,k);

model.W = W;
model.predict = @predict;
end

function [f,g] = softMax(w,X,y,k)
[n,d] = size(X);
k = max(y);

% Reshape w's dimensions to "d x k"
% d = number of features
% k = number of classes
W = reshape(w, [d k]);

% XW = (X*W);
% 
% size(y)
% size(XW)
% size(log(sum(exp(XW), 2))) % 1 x 5
% size(exp(XW)) % 500 x 5
% size(sum(exp(XW), 2)) % 500 x 1
% 
% a = (-W' * X');
% size(a')
% b = log(sum(exp(XW), 2));
% size(b)

% size(W'*X')
% -W .* X
% X .* W

% f = sum(log(1 + exp(-yXW)));    % Function value
% g = -X'*(y./(1+exp(yXW)));      % Gradient

% Compute loss
f = 0;
for i = 1:n
    yi = y(i);
    term1 = -W(:,yi)' * X(i);
    term2 = 0;
    for c = 1:k
        term2 = term2 + exp(W(:,c)' * X(i));
    end
    f = term1 + term2 + f;
end
f

fprintf('hi');

% Compute gradient
g = zeros(d,k);
for c = 1:k
    eachK = zeros(n,d);
    for i = 1:n
        denom = 0;
        for cPrime = 1:k
            denom = denom + exp(W(:,cPrime)' * X(i));
        end
        
        for j = 1:d
            eachK(i,j) = (-X(i,j) * (y(i) == c) + exp(W(:,c)' * X(i)) * X(i) / denom);
        end
    end
    g(:,c) = sum(eachK);
end

% Reshape the gradient's dimensions to a 1-D vector
% "1 x (d * k)"

g = g(:)
% g = reshape(g, [d*k 1]);

fprintf('f is %d with size %i x %i\n', f, size(f,1), size(f,2));
% size(f,1)
% size(f,2)
% size(g)
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end