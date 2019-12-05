%% 
% Inputs: 
%           data=matrix with the data and classes. First column contains x
%           coordinates, second column the y coordinates and the third
%           column the class (only two classes, 0 and 1)
%           K=number classes
%           idx= cluster indices
%           C=cluster centroid location
% Outputs:
%           subplot of 3 columns with original data, K-means clustering and
%           centroids
%
% Author:
%           Pietro Gori 
function [] = plotResultsKMeans(data,K,idx,C)

    x1 = min(data(:,1)):0.01:max(data(:,1));
    x2 = min(data(:,2)):0.01:max(data(:,2));
    [x1G,x2G] = meshgrid(x1,x2);
    XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
    idx2Region = kmeans(XGrid,K,'MaxIter',1,'Start',C);
    
    f=figure;
    %set(f,'Position',[527         186        1307         333])
    subplot(1,3,1)
    hold on
    plot(data(data(:,3)==0,1),data(data(:,3)==0,2),'bx','MarkerSize',7,'LineWidth',2)
    plot(data(data(:,3)==1,1),data(data(:,3)==1,2),'ro','MarkerSize',7,'LineWidth',2)   
    title('Original data','FontSize',16,'FontWeight','bold')
    legend('class 1','class 2','Location','best');  

    subplot(1,3,2)
    hold on
    gscatter(XGrid(:,1),XGrid(:,2),idx2Region);
    plot(data(data(:,3)==0,1),data(data(:,3)==0,2),'kx','MarkerSize',7,'LineWidth',2)
    plot(data(data(:,3)==1,1),data(data(:,3)==1,2),'ko','MarkerSize',7,'LineWidth',2)
    title('K-means clustering','FontSize',16,'FontWeight','bold')
    hold off


    list_colors=['b','r','k','m','p'];
    if K>length(list_colors)
        error('Please add other colors')
    end
    subplot(1,3,3)
    hold on
    for i=1:K
        plot(data(idx==i,1),data(idx==i,2),[list_colors(i) 'x'],'MarkerSize',7,'LineWidth',2)
    end    
    plot(C(:,1),C(:,2),'go','MarkerSize',12,'LineWidth',2,'DisplayName','centroid')
    hold off
    title('K-means centroids','FontSize',16,'FontWeight','bold')
    legend('AutoUpdate','off','Location','best')

end