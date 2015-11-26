function [model] = recommendSVD(X,y,k)
n = max(X(:,1));
d = max(X(:,2));
nRatings = size(X,1);

% Initialize parameters
% - for the biases, we'll use the user/item averages
% - for the latent factors, we'll use small random values
subModel = recommendUserItemMean(X,y);
bu = subModel.bu/2;
bm = subModel.bm/2;
W = .00001*randn(k,d);
Z = .00001*randn(n,k);

% Optimization
maxIter = 10;
alpha = 1e-2;
for iter = 1:maxIter
    for j = 1:nRatings
        % Compute gradient
        gu = zeros(n,1);
        gm = zeros(d,1);
        gW = zeros(k,d);
        gZ = zeros(n,k);
    
        % Pick one sample randomly
        i = randi([1,nRatings]);

        % Make prediction for this rating based on current model
        u = X(i,1);
        m = X(i,2);
        yhat = bu(u) + bm(m) + W(:,m)'*Z(u,:)';

        % Use the gradient for the random sample
        r = y(i)-yhat;
        gu(u) = - r;
        gm(m) = - r;
        gW(:,m) = - r*Z(u,:)';
        gZ(u,:) = - r*W(:,m)';

        % Take a small step in the negative gradient directions
        bu = bu - alpha*gu;
        bm = bm - alpha*gm;
        W = W - alpha*gW;
        Z = Z - alpha*gZ;
    end

    % Compute and output function value
    f = 0;
    for i = 1:nRatings
        u = X(i,1);
        m = X(i,2);
        yhat = bu(u) + bm(m) + W(:,m)'*Z(u,:)';
        f = f + (1/2)*(y(i) - yhat)^2;
    end
    fprintf('Iter = %d, f = %e\n',iter,f);
end

model.bu = bu;
model.bm = bm;
model.W = W;
model.Z = Z;
model.predict = @predict;
end

function [y] = predict(model,X)
t = size(X,1);
bu = model.bu;
bm = model.bm;
W = model.W;
Z = model.Z;

y = zeros(t,1);
for i = 1:t
    u = X(i,1);
    m = X(i,2);
    y(i) = bu(u) + bm(m) + W(:,m)'*Z(u,:)'; % Take the average between user and movie ratings
end
end