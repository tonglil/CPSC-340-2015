function [w,f] = findMinL1(funObj,w,lambda,maxEvals,verbose,varargin)
% Find local minimizer of differentiable function plus lambda*norm(w,1)

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g] = funObj(w,varargin{:});
funEvals = 1;

alpha = min(1,1/sum(abs(g)));
while 1
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*g; % Gradient step
    w_new = sign(w_new).*max(abs(w_new)-lambda*alpha,0); % Proximal step
	[f_new,g_new] = funObj(w_new,varargin{:});
	funEvals = funEvals+1;
    
    gtd = g'*(w_new-w);
    while f_new + lambda*sum(abs(w_new)) > f + lambda*sum(abs(w)) + gamma*alpha*gtd
        if verbose
            fprintf('Backtracking...\n');
        end
        alpha = alpha/2;
        w_new = w - alpha*g; % Gradient step
        w_new = sign(w_new).*max(abs(w_new)-lambda*alpha,0); % Proximal step
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
	optCond = norm(w-sign(w-g).*max(abs(w-g)-lambda,0),'inf');
    if verbose
        fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alpha,f+lambda*sum(abs(w)),optCond);
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
    
    
   %% Update step-size for next iteration
    if ~isLegal(alphaBB) || alphaBB < 1e-10 || alphaBB > 1e10
        fprintf('Resetting\n');
       alpha = 1; 
    else
        alpha = alphaBB;
    end
end