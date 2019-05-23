function [P] = color2position(C,Min,Max)
% COLOR2POSITION  Converts a given color vector or image into 3-dimensional
% coordinates.
%   P = COLOR2POSITION(C) converts the input C from color to 3-dimensional
%   coordinates. Input C could either be in the shape of a nx3 color vector
%   or a mxnx3 image.
%   P = COLOR2POSITION(C,Min) converts the input C using Min as the minimum
%   coordinate.
%   P = COLOR2POSITION(C,Min,Max) converts the input C using Min and Max as
%   the boundary coordinates.
if( ~isdouble(C) )
    C = color2double(C);
end
if( nargin < 3 )
    Max = ones(1,3);
end
if( nargin < 2 )
    Min = -ones(1,3);
end
if( ndims(C) == 3 )
    Min = reshape(Min,1,1,3);
    Max = reshape(Max,1,1,3);
end
P = Min + C .* (Max-Min);
end