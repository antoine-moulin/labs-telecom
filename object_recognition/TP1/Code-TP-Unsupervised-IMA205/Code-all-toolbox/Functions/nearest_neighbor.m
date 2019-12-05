%% Nearest Neighbour
% Inputs: 
%           Btrain: Scores of the training set
%
%           Id_Train: Identification subject number of training set
%
%           Btest: Scores of the test set
%
%           Id_Test: Identification subject number of test set
%
% Outputs:
%           result: Percentage of correct answer
%
%           correct: vector of the same size of the test set containing 1
%           if the image has been correctly assigned and 0 otherwise
%
% Author:
%           Pietro Gori 

function [result,correct] = nearest_neighbor(Btrain,Id_Train,Btest,Id_Test)
    
    [Ntrain,d]=size(Btrain);
    [Ntest,dbis]=size(Btest);

    if d~=dbis
        error('Error')
    end    
    
    NormTrain2 = repmat(sqrt(sum(Btrain.^2, 2))',Ntest,1);
    NormTest2 = repmat(sqrt(sum(Btest.^2, 2)),1,Ntrain);
    Inner=Btest*Btrain';
    angle=acos(Inner./(NormTrain2.*NormTest2));     
    

    % Test subject is assigned to the training subject with the minimal
    % angle
    Id_calc=zeros(Ntest,1);
    for i=1:Ntest
        [~,Ind] = min(angle(i,:));
        Id_calc(i)=Id_Train(Ind);    
    end

    correct=(Id_calc-Id_Test)==0;
    result=sum(correct)*100/Ntest;
end