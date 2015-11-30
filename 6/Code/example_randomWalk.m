load simpleGraph.mat % Loads adjacency A and labelList

n = length(A);
p = zeros(n,2);
r = 100;

for i = 1:n
    for j = 1:r
        % Run random walk
        yhat = runRandomWalk(A,labelList,i);
        if yhat == 1
            p(i,1) = p(i,1) + 1;
        elseif yhat == -1
            p(i,2) = p(i,2) + 1;
        end
    end
end

% Output final probabilities
probabilities = p/r