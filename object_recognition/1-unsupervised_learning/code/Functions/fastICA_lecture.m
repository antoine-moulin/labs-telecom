%% %%%%%%%%%%%%%%%%%%%%% FAST ICA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               
% Inputs: 
%               X: is a [d x N] matrix. Every column is an observation 
%               and every row is a feature.       
%
%               (Optional) N_Iter: maximum number of iterations
%
%               (Optional) delta: convergence criteria threshold
%
%               (Optional) plot_evolution: plot evolution of error
%               
% Outputs:      
%               S: [d x N] matrix.  Each column is an independent component 
%               of the centred and whitened input data X              
%               
%               W: [d x d] matrix. It is the demixing matrix. S = W*Xcw 
%
%               WXc: [d x d] matrix. It is the demixing matrix of the
%               centred (and not whitened) data
%
% Author:       
%               Pietro Gori
%
% References:   
%               1- Aapo Hyvarinen and Erkki Oja - Independent Component 
%                  Analysis: A Tutorial - 1999
%               2- Tichavsk et al. - Performance Analysis of the FastICA
%                  Algorithm - IEEE TSP - 2006               

function [S, W, WXc] = fastICA_lecture(X,N_Iter,delta,plot_evolution)

if nargin < 1
    error('ICA - Minimum number of inputs is 1')
end

if nargin == 1
    MAX_ITERS = 3000;    % Max # iterations
    TOL = 1e-5;         % Convergence criteria    
    plot_evolution=0;
    
elseif nargin == 2
    if isempty(N_Iter)
        N_Iter=3000;
    end
    MAX_ITERS = N_Iter; 
    TOL = 1e-5;     
    plot_evolution=0;
    
elseif nargin == 3
    if isempty(N_Iter)
        N_Iter=3000;
    end
    if isempty(delta)
        delta=1e-5;
    end
    MAX_ITERS = N_Iter; 
    TOL = delta;  
    plot_evolution=0;
elseif nargin == 4
    if isempty(N_Iter)
        N_Iter=3000;
    end
    if isempty(delta)
        delta=1e-5;
    end
    MAX_ITERS = N_Iter; 
    TOL = delta;  
    if isempty(plot_evolution)
        plot_evolution=0;
    end
end   

% Size
[d,N]=size(X);

% Compute sample mean
mu = mean(X,2);

% Subtract mean
Xc = X - repmat(mu,1,N); % X - (X*ones(N,1)*ones(1,N))/N;

% Covariance matrix
C=(Xc*Xc')/(N-1);

% Whiten data
Xcw=sqrtm(C)\Xc; % (Z*Z')/(N-1);

% Initialize W
W=orth(randn(d,d)); % random orthogonal matrix 

k = 0;
delta = inf;

if plot_evolution==1
    figure
    hold on         
    title('Error evolution ICA')
end

while delta > TOL && k < MAX_ITERS
    
    k = k + 1;
    W_old = W;
    
    Wp = g(W*Xcw)*Xcw' - diag(gp(W*Xcw)*ones(N,1))*W;
    W = sqrtm(Wp * Wp')\Wp; % W*W'=I
    
    delta = 1-min(abs(diag(W'*W_old)));
    
    if rem(k,100)==0
        disp(['Iteration ICA number ' num2str(k) ' out of ' num2str(MAX_ITERS) ', delta = ' num2str(delta) ])
    end
    
    if plot_evolution==1
        plot(k,delta,'bx')            
    end

end

if k==MAX_ITERS
    disp(['Maximum number of iterations reached ! delta = ' num2str(delta) ])
else
    disp(['Convergence achieved ( delta = ' num2str(delta) ') in ' num2str(k) ' iterations'])
end

% Independent components
S = W * Xcw;

% Compute W in the space of the centred X
WXc=W/sqrtm(C);

end

% 
% function res = G(u)
%     res = -exp(-0.5 * u.^2);
% end

function res = g(u)
    res = u .* exp(-0.5 * u.^2);
end

function res = gp(u)
    res = (1-u.^2) .* exp(-0.5 * u.^2);
end


