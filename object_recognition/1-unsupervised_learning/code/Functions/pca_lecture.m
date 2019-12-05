%% 
% Inputs: 
%           X: is a [Nxd] matrix. Every row is an observation and every
%           column consists of features.
%
% Outputs:
%           Y: is a [Nxd] matrix representing the scores, namely the 
%           coordinates of X onto the new basis given by the eigenvactors U
%           of the covariance matrix of X. Columns are the principal components.
%           
%           U: columns are Eigenvectors (sorted from the greatest to the lowest eigenvalue)
%
%           D: Eigenvalues (sorted from the greatest to the lowest eigenvalue)
%           
%           var_explained: percentage of the original variability explained
%           by each principal component.
%
% Author:
%           Pietro Gori 

function [Y,U,D,var_explained] = pca_lecture(X)
    N=size(X,1); 
    Xc=X-repmat(mean(X),N,1); % centering
    [~,D2,U] = svd(Xc,'econ'); 
    D2=diag(D2);
    Y=Xc*U; % computation of the scores
    D=D2.^2./(N-1); % compute eigenvalues
    tot=sum(D);
    var_explained=D*100/tot;
end