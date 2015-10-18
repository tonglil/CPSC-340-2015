function [model] = decisionTree(X,y,maxDepth)
% [model] = decisionStump(X,y)
%
% Fits a decision tree that splits on a sequence of single variables,
%   assuming that X is binary {0,1}, and y is categorical {1,2,3,...,C}.

[N,D] = size(X);

% Learn a decision stump
splitModel = decisionStump_InfoGain(X,y);

if maxDepth <= 1 || isempty(splitModel.splitVariable)
    % If we have reached the maximum depth or the decision stump does
    % nothing, use the decision stump
    model = splitModel;
else
    % Fit a decision tree to each split, decreasing maximum depth by 1
    d = splitModel.splitVariable;
    model.splitModel = splitModel;
    
    % Find indices of examples in each split
    splitIndex1 = find(X(:,d)==1);
    splitIndex0 = find(X(:,d)==0);
    
    % Fit decision tree to each split
    model.subModel1 = decisionTree_InfoGain(X(splitIndex1,:),y(splitIndex1),maxDepth-1);
    model.subModel0 = decisionTree_InfoGain(X(splitIndex0,:),y(splitIndex0),maxDepth-1);
    
    % Assign prediction function
    model.predictFunc = @predict;
end
end

function [y] = predict(model,X)
[T,D] = size(X);
y = zeros(T,1);

% Predict based on first split
splitModel = model.splitModel;
yhat = splitModel.predictFunc(splitModel,X);

if isempty(splitModel.splitVariable)
    % If no further splitting, return the majority label
    y = splitModel.label1*ones(T,1);
else
    % Recurse on both sub-models
    d = splitModel.splitVariable;
    
    splitIndex1 = find(X(:,d)==1);
    splitIndex0 = find(X(:,d)==0);
    
    subModel1 = model.subModel1;
    subModel0 = model.subModel0;
    
    y(splitIndex1) = subModel1.predictFunc(subModel1,X(splitIndex1,:));
    y(splitIndex0) = subModel0.predictFunc(subModel0,X(splitIndex0,:));
end
end