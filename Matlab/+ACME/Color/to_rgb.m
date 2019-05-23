function [ RGB ] = to_rgb( V, range )
% TO_RGB  Converts the given values into RGB format
%   RGB = TO_RGB(V) returns a nx3 color vector from the nx1 vector V.
%   RGB = TO_RGB(V, range) rescale the value in range before the
%   conversion.
if( nargin < 2 )
range = [min(V) max(V)];
end
range = sort(range);
C     = clamp(normalize(V,range(1),range(2)),range(1),range(2));
RGB   = hsv2rgb(value2hsv(C));
end