function [yhat] = runRandomWalk(A,labelList,i)
while 1
    neighbors = find(A(i,:)==1);

    if any(labelList(:,1) == i)
        % Case 2: i is one of the labeled nodes
        degree = size(neighbors,2);
        probability = 1 / (degree + 1);
        
        if rand < probability
            index = find(labelList(:,1) == i);
            yhat = labelList(index,2);
            return;
        end
    end
    
    % Case 1: i is not one of the labeled nodes
    % Or
    % Case 2: not returning the label due to probability
    i = randsample(neighbors,1);
end
end