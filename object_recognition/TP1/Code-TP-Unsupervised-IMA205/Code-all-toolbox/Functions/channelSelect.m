function [Igray] = channelSelect(I, channel)
%CHANNELSELECT convert I to a grayscale image in the desired channel
%   [ Igray ] = channelSelect( I, channel )
%   channel contains a string : 
%   'meanRGB' average the channels RGB
%   'r' or 'red' to select the red channel in RGB space
%   'g' or 'green' to select the green channel in RGB space
%   'b' or 'blue' to select the blue channel in RGB space
%   'X' to select the X channel in CIE-XYZ space

 % selecting meanRGB : averaging the channels R, G, B
    if strcmp(channel,'meanRGB')
        Igray = sum(I,3)/3;
        
    % selecting red channel
    elseif strcmp(channel, 'r') || strcmp(channel, 'red')
        Igray = I(:,:,1);
        
    % selecting green channel
    elseif strcmp(channel, 'g') || strcmp(channel, 'green')
        Igray = I(:,:,2);
    
    % selecting blue channel
    elseif strcmp(channel, 'b') || strcmp(channel, 'blue')
        Igray = I(:,:,3);
    
    % selecting V channel from HSV
    elseif strcmp(channel, 'v') || strcmp(channel, 'V')
        Igray = max(I,[],3);
    
    % selecting X channel from CIE-XYZ
    elseif strcmp(channel,'X')
        cR = 0.4125;
        cG = 0.3576;
        cB = 0.1804;
        Igray = cR*I(:,:,1) + cG*I(:,:,2) + cB*I(:,:,3);
    
    % selecting meanRGB by default
    else 
        warning('non existent or invalid channel argument : assumed meanRGB')
        Igray = sum(I,3)/3;
    end

end
