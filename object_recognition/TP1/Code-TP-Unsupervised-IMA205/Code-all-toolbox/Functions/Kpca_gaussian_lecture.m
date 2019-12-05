%% Gaussian Kernel - PCA
% Inputs: 
%           X: is a [Nxd] matrix. Every row is an observation and every
%           column is a feature.
%
% Outputs:
%           Y: is a [Nxd] matrix representing the scores, namely the 
%           coordinates of \phi(X) onto the new basis given by the eigenvactors 
%           of the covariance matrix of \phi(X). Columns are the principal components.
%
%           An: columns are Eigenvectors normalised (sorted from the greatest
%           to the lowest eigenvalue)
%
%           D: Eigenvalues (sorted from the greatest to the lowest eigenvalue)
%           
%           var_explained: percentage of the original variability explained
%           by each principal component.
%
% Author:
%           Pietro Gori
%
% References: 
%           B. Scholkopf et al. - Nonlinear component analysis as a kernel
%           eigenvalue problem - Neural Computation - 1998

function [Y,An,D,var_explained] = Kpca_gaussian_lecture(X,sigma)

    N=size(X,1); 
    K = kernel_matrix(X,sigma); % kernel construction
    
    oneN=ones(N,N)/N;
    Kc=K-oneN*K-K*oneN+oneN*K*oneN; % center kernel matrix

    %% eigenvalue analysis
    [A,D]=eig(Kc);    
    D=diag(D); 
    [D,I]=sort(D,'descend');
    A=A(:,I);
    
    % eigenvalue and variance explained  
    tot=sum(D);
    var_explained=D*100/tot;
    
    % Normalisation eigenvectors
    % Norm of every eigenvector is 1, we want it to be 1/sqrt(N*eig)
    
    An=A;
    for i=1:N        
        An(:,i)=A(:,i)*(1/sqrt((N-1)*D(i)));             
    end       
    
    Y=Kc*An; % computation of the scores   

end

function K = kernel_matrix(X,sigma)
    
    N=size(X,1);

    NormX2 = repmat(sum(X.^2, 2),1,N);
    NormY2 = repmat(sum(X.^2, 2)',N,1);
    InnerX = X * X';
    D = NormX2+NormY2-2*InnerX;
    D(D<1e-10)=0;    
    
    K=exp(-D./(2*sigma^2));

end