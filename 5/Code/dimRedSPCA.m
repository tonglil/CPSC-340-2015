function [model] = dimRedSPCA(X,k)
[n,d] = size(X);
lambda = sqrt(n*d);

% Subtract mean
mu = mean(X);
X = X - repmat(mu,[n 1]);

% Initialize W and Z
W = randn(k,d);
Z = randn(n,k);

f = (1/2)*sum(sum((X-Z*W).^2));
for iter = 1:50
    fOld = f;
    
    % Update Z
    Z(:) = findMinL1(@funObjZ,Z(:),lambda,10,0,X,W);
    
    % Update W
    W(:) = findMinL1(@funObjW,W(:),lambda,10,0,X,Z);
    
    f = (1/2)*sum(sum((X-Z*W).^2));
    fprintf('Iteration %d, loss = %.5e\n',iter,f);
    
    if fOld - f < 1
        break;
    end
end

model.k = k;
model.mu = mu;
model.W = W;
model.compress = @compress;
model.expand = @expand;
end

function [Z] = compress(model,X)
[t,d] = size(X);
mu = model.mu;
W = model.W;

X = X - repmat(mu,[t 1]);

[n,~] = size(X);

% Find Z minimizing the object with L1-regularization

Z = randn(n,model.k);
lambda = sqrt(n*d);

f = (1/2)*sum(sum((X-Z*W).^2));
for iter = 1:50
    fOld = f;
    
    % Update Z
    Z(:) = findMinL1(@funObjZ,Z(:),lambda,10,0,X,W);
    
    f = (1/2)*sum(sum((X-Z*W).^2));
    fprintf('Iteration %d, loss = %.5e\n',iter,f);
    
    if fOld - f < 1
        break;
    end
end
end

function [X] = expand(model,Z)
[t,d] = size(Z);
mu = model.mu;
W = model.W;

X = Z*W + repmat(mu,[t 1]);
end

function [f,g] = funObjW(W,X,Z)
% Resize vector of parameters into matrix
d = size(X,2);
k = size(Z,2);
W = reshape(W,[k d]);

% Compute function and gradient
R = X-Z*W;
f = (1/2)*sum(sum(R.^2));
g = -Z'*R;

% Return a vector
g = g(:);
end

function [f,g] = funObjZ(Z,X,W)
% Resize vector of parameters into matrix
n = size(X,1);
k = size(W,1);
Z = reshape(Z,[n k]);

% Compute function and gradient
R = X-Z*W;
f = (1/2)*sum(sum(R.^2));
g = -(R*W');

% Return a vector
g = g(:);
end
