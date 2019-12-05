%% 
% Inputs: 
%           r1=radius first circle
%           steps1=angle step for first circle
%           r2=radius second circle
%           step2=angle step for second circle
%
% Outputs:
%           data=matrix with the data and classes. First column contains x
%           coordinates, second column the y coordinates and the third
%           column the class (only two classes, 0 and 1)
%
% Author:
%           Pietro Gori 
function data = create_circles(r1,step1,r2,step2)

    % 1 circle
    angle1=0:step1:2*pi;
    x1=r1*cos(angle1);
    y1=r1*sin(angle1);
    
    % 2 circle
    angle2=0:step2:2*pi;
    x2=r2*cos(angle2);
    y2=r2*sin(angle2);

    data=[x1' y1' zeros(length(x1),1); x2' y2' ones(length(x2),1)];

end