%% Non-negative matrix factorization
% Inputs: 
%           X: is a [dxN] matrix. Every column (x) is an observation and every
%           row consists of features.
%
%           r: size of the matrices W and H
%
%           (Optional) N_Iter: maximum number of iterations
%
%           (Optional) tolerance: convergence criteria threshold
%
%           (Optional) plot_evolution: plot evolution convergence criteria
%
% Outputs:
%           W: is a [d x r] matrix containing the basis images in its
%           columns
%           
%           H: is a [r x N] matrix containing the loadings (h) in its columns
%           of the linear combination: x=Wh 
%
% Author:
%           Pietro Gori 
%
% References:
%           Lee et al. - Algorithms for Non-negative Matrix Factorization

function [W,H] = nnmf_lecture(X,r,N_Iter,tolerance,plot_evolution)

    if isempty(N_Iter)
        N_Iter=1000;
    end
    if isempty(tolerance)
        tolerance=1e-5;
    end
    if isempty(plot_evolution)
        plot_evolution=0;
    end

    % Test for positive values
    if min(min(X)) < 0
        error('Input matrix X has negative values !');        
    end

    % Size
    [d,N]=size(X);
   
    % Initialization
    W=rand(d,r);
    H=rand(r,N);       
    
    % parameters for convergence
    k = 0;
    delta = inf;
 
    if plot_evolution==1
        figure
        hold on     
        title('Error evolution NNMF')
    end

    while delta > tolerance && k < N_Iter
        
        % multiplicative method   
        H = H.*(W'*X)./((W'*W)*H + eps); 
        W = W.*(X*H')./(W*(H*H') + eps);

        % Convergence indices
        k = k + 1;            
%         delta = sqrt( (sum(sum((X-W*H).^2))) ) / sqrt( (sum(sum(X.^2))) ); % |X-WH|_2 / |X|_2
        diff=X-W*H;        
        delta=norm(diff,'fro') / norm(X,'fro'); % sqrt(trace(diff'*diff)) / sqrt(trace(X'*X))

        if rem(k,100)==0
            disp(['Iteration NNMF number ' num2str(k) ' out of ' num2str(N_Iter)...
                ', delta= ' num2str(delta)...
                ', eucl error: ' num2str(sum(sum((diff).^2)))...
                ', frobenius norm: ' num2str(norm(diff,'fro')) ])
        end
        
        if plot_evolution==1
            plot(k,delta,'bx')            
        end
        

    end

    if k==N_Iter
        disp(['Maximum number of iterations reached ! delta = ' num2str(delta) ])
    else
        disp(['Convergence achieved ( delta = ' num2str(delta) ') in ' num2str(k) ' iterations'])
    end
end



