%% 
% Inputs: 
%           data=matrix with the data and classes. First column contains x
%           coordinates, second column the y coordinates and the third
%           column the class (only two classes, 0 and 1)
%           Y=scores of PCA
%           U= eigenvectors PCA
%           type=string with the type of PCA (linear or K-PCA)
% Outputs:
%           subplot of 3 columns with original data, first and second PC
%
% Author:
%           Pietro Gori 
function [] = plotResultsPCA(data,Y,U,type)

if nargin < 2
    error('Minimum number of inputs is 2')
end

if nargin < 4    
    type='';
end


L0=sum(data(:,3)==0);
L1=sum(data(:,3)==1);

f=figure;
%set(f,'Position',[522        1017        1307         333])
subplot(1,3,1)
title('Original data','FontSize',16,'FontWeight','bold')
hold on
plot(data(data(:,3)==0,1),data(data(:,3)==0,2),'bx','MarkerSize',7,'LineWidth',2)
plot(data(data(:,3)==1,1),data(data(:,3)==1,2),'ro','MarkerSize',7,'LineWidth',2)
if ~isempty(U)
    average=mean(data(:,1:2));
    sd=norm(std(data(:,1:2)));
    u1=U(:,1)*sd;
    u2=U(:,2)*sd;
    if abs(u1'*u2)>1e-5
        error('Error. Eigenvectors should be orthogonal')
    end
    plot([average(1)-u1(1) average(1)+u1(1)],[average(2)-u1(2) average(2)+u1(2)],'g-','linewidth',3)
    plot([average(1)-u2(1) average(1)+u2(1)],[average(2)-u2(2) average(2)+u2(2)],'k-','linewidth',3)
    legend('class 1','class 2','PC1','PC2','Location','best');
else
    legend('class 1','class 2','Location','best');
end

subplot(1,3,2)
hold on
plot(Y(data(:,3)==0,1),zeros(L0,1),'bx')
plot(Y(data(:,3)==1,1),zeros(L1,1),'ro')
if isempty(type)
    title('First PC','FontSize',16,'FontWeight','bold')
else
    title(['First PC (' type ')'],'FontSize',16,'FontWeight','bold')
end
legend('class 1','class 2','Location','best');

subplot(1,3,3)
hold on
plot(Y(data(:,3)==0,2),zeros(L0,1),'bx')
plot(Y(data(:,3)==1,2),zeros(L1,1),'ro')
if isempty(type)
    title('Second PC','FontSize',16,'FontWeight','bold')
else
    title(['Second PC (' type ')'],'FontSize',16,'FontWeight','bold')
end
legend('class 1','class 2','Location','best');

end