function [w,f] = findMin(funObj,w,maxEvals,verbose,varargin)
% Find local minimizer of differentiable function

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g] = funObj(w,varargin{:});
funEvals = 1;

alpha = min(1,1/sum(abs(g)));
while 1
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*g;
	[f_new,g_new] = funObj(w_new,varargin{:});
	funEvals = funEvals+1;
    
    gg = g'*g;
    while f_new > f - gamma*alpha*gg
        if verbose
            fprintf('Backtracking...\n');
        end
        alpha = max(alpha^2*gg/(2*(f_new - f + alpha*gg)),alpha*1e-3);
        w_new = w - alpha*g;
        [f_new,g_new] = funObj(w_new,varargin{:});
        funEvals = funEvals+1;
    end
    
    % Guess of step-size for next iteration
    s = w_new - w;
    y = g_new - g;
    alphaBB = (s'*s)/(s'*y);
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
	
    %% Test termination conditions
	optCond = norm(g,'inf');
    if verbose
        fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alpha,f,optCond);
    end
	
	if optCond < optTol
        if verbose
            fprintf('Problem solved up to optimality tolerance\n');
        end
		break;
	end
	
	if funEvals >= maxEvals
        if verbose
            fprintf('At maximum number of function evaluations\n');
        end
		break;
    end
    
    if alpha < 1e-10
        if verbose
            fprintf('Step size has gotten too small\n');
        end
		break;
    end
    
    %% Update step-size for next iteration
    if ~isLegal(alphaBB) || alphaBB < 1e-10 || alphaBB > 1e10
        if verbose
            fprintf('Resetting\n');
        end
       alpha = 1; 
    else
        alpha = alphaBB;
    end
end