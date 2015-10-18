function [model] = naiveBayes(X,y)
% [model] = naiveBayes(X,y)
%
% Implementation of navie Bayes classifier for binary features

% Compute number of training examples and number of features
[N,D] = size(X);
% N: number of examples
% D: number of features

% Computer number of class lables
C = max(y);

% Number of occurrences of each label
counts = zeros(C,1);
for c = 1:C
    counts(c) = sum(y==c);
end

p_y = counts/N; % This is the probability of each class, p(y(i) = c)

% We will store:
%   p(x(i,j) = 1 | y(i) = c) as p_xy(j,1,c)
%   p(x(i,j) = 0 | y(i) = c) as p_xy(j,2,c)

% The conditional probability of values based on frequency in data set
p_xy = zeros(D,2,C);

wordlistIndexes = zeros(4,4);

for c = 1:C
    y_i = y==c;         % where y(i) = c
    Xij = X(y_i,:);     % Xs where y(i) = c
    Xij1 = sum(Xij==1); % Xs where x(i,j) = 1 | y(i) = c
    Xij0 = sum(Xij==0); % Xs where x(i,j) = 0 | y(i) = c
    probXij1 = Xij1/counts(c);  % probabilities based on frequency
    probXij0 = Xij0/counts(c);
    
    % Get the top 3 popular words
    [~,I] = sort(probXij1, 'descend');
    wordlistIndexes(c,:) = [c,I(1:3)];
    
    p_xy(:,1,c) = probXij1;
    p_xy(:,2,c) = probXij0;
end

model.C = C;
model.p_y = p_y;
model.p_xy = p_xy;
model.predict = @predict;
model.wordlistIndexes = wordlistIndexes;
end

function [yhat] = predict(model,Xtest)
[T,D] = size(Xtest);
C = model.C;
p_y = model.p_y;
p_xy = model.p_xy;

yhat = zeros(T,1);
for i = 1:T
    probs = p_y; % This will be the probability for each class
    for j = 1:D
        if Xtest(i,j) == 1
            for c = 1:model.C
                probs(c) = probs(c)*p_xy(j,1,c);
            end
        else
            for c = 1:model.C
                probs(c) = probs(c)*p_xy(j,2,c);
            end
        end
    end
    [maxProb,yhat(i)] = max(probs);
end
end