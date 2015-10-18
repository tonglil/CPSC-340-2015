function [errorAverage] = folds(Xfold1, Xfold2, y1, y2, depth)
    % train on fold 1
    [N1,D] = size(Xfold1);
    model1 = decisionTree_InfoGain(Xfold1,y1,depth);
    
    yhat1 = model1.predictFunc(model1,Xfold2);
    errorTrain1 = sum(yhat1 ~= y2)/N1;
    
    % train on fold 2
    [N2,D] = size(Xfold1);
    model2 = decisionTree_InfoGain(Xfold2,y2,depth);
    
    yhat2 = model2.predictFunc(model2,Xfold1);
    errorTrain2 = sum(yhat2 ~= y1)/N2;
    
    errorAverage = mean([errorTrain1, errorTrain2])
end