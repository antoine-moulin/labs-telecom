%% 
% Inputs: 
%           mu1=average first Gaussian distribution
%           sigma1=std first Gaussian distribution
%           L1=number points first Gaussian distribution
%           mu2=average second Gaussian distribution
%           signam2=std second Gaussian distribution
%           L2=number points second Gaussian distribution
%
% Outputs:
%           data=matrix with the data and classes. First column contains x
%           coordinates, second column the y coordinates and the third
%           column the class (only two classes, 0 and 1)
%
% Author:
%           Pietro Gori
function data = create_gauss(mu1,sigma1,L1,mu2,sigma2,L2)

    % gauss 1
    r1 = mvnrnd(mu1,sigma1,L1);
    % gauss 2
    r2 = mvnrnd(mu2,sigma2,L2);
    data=[r1 zeros(L1,1); r2 ones(L2,1)];

end