function [model] = logRegL0(X,y,lambda)
[n,d] = size(X);
maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 0; % Whether or not to display progress of algorithm
w0 = zeros(d,1);
oldScore = inf;

% Fit model with only 1 variable, 
% and record 'score' which is the loss plus the regularizer
ind = 1;
w = findMin(@logisticLoss,w0(ind),maxFunEvals,verbose,X(:,ind),y);
score = logisticLoss(w,X(:,ind),y) + lambda*length(w);
minScore = score;
minInd = ind;

while minScore ~= oldScore
    oldScore = minScore;
    fprintf('\nCurrent set of selected variables (score = %f):',minScore);
    fprintf(' %d',ind);
    
    for i = 1:d
        if any(ind == i)
            % This variable has already been added
            continue;
        end
        
        % Fit the model with 'i' added to the features,
        % then compute the score and update the minScore/minInd
        newInd = union(ind,i);
        
        w = findMin(@logisticLoss,w0(newInd),maxFunEvals,verbose,X(:,newInd),y);
        score = logisticLoss(w,X(:,newInd),y) + lambda*length(w);
        
        if score < minScore
            minScore = score;
            minInd = newInd;
        end
    end
    ind = minInd;
end

model.w = zeros(d,1);
model.w(minInd) = findMin(@logisticLoss,w0(minInd),maxFunEvals,verbose,X(:,minInd),y);
model.predict = @(model,X)sign(X*model.w); % Predictions by taking sign
end

function [f,g] = logisticLoss(w,X,y)
yXw = y.*(X*w);
f = sum(log(1 + exp(-yXw))); % Function value
g = -X'*(y./(1+exp(yXw))); % Gradient
end