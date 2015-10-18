
% Load data
warning off all
close all
clear all
load nonLinearData.mat
[n,d] = size(X);

% Plotting Code
plot(X,y,'b.');hold on
plot(Xtest,ytest,'g.');
xl = xlim;
yl = ylim;
Xvals = [xl(1):.1:xl(2)]';
pause(.1)

% Display result of fitting with RBF kernel
for sigma = 2.^[3:-1:-4]
    %% Train on X, test on Xtest
    model = leastSquaresRBF(X,y,sigma);
    yhat = model.predict(model,Xtest);
    fprintf('Test error with sigma = %f is %f\n',sigma,mean(abs(yhat-ytest)));
    
    %% Plotting Code
    figure(1);clf;
    plot(X,y,'b.');hold on
    plot(Xtest,ytest,'g.');
    yvals = model.predict(model,Xvals);
    plot(Xvals,yvals,'r-');
    legend({'Train','Test'});
    ylim(yl);
    title(sprintf('RBF Basis (sigma = %f)',sigma));
    pause(.25)
    
end