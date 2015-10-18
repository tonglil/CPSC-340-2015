function [w,f] = findMin(funObj,w,maxEvals,varargin)

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g] = funObj(w,varargin{:});
funEvals = 1;

alpha = 1;
while 1
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*g;
	[f_new,g_new] = funObj(w_new,varargin{:});
	funEvals = funEvals+1;
    
    gg = g'*g;
    while f_new > f - gamma*alpha*gg
        fprintf('Backtracking...\n');
        alpha = alpha^2*gg/(2*(f_new - f + alpha*gg));
        w_new = w - alpha*g;
        [f_new,g_new] = funObj(w_new,varargin{:});
        funEvals = funEvals+1;
    end

    %% Update step-size for next iteration
    y = g_new - g;
    alpha = -alpha*(y'*g)/(y'*y);
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
	
    %% Test termination conditions
	optCond = norm(g,'inf');
	fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alpha,f,optCond);
	
	if optCond < optTol
        fprintf('Problem solved up to optimality tolerance\n');
		break;
	end
	
	if funEvals >= maxEvals
        fprintf('At maximum number of function evaluations\n');
		break;
	end
end