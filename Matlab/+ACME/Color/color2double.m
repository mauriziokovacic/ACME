function [C] = color2double(C)
% COLOR2DOUBLE  Converts a given color from range [0-255] to [0-1].
%   C = COLOR2DOUBLE(C) returns a nx3 vector.
if(~isdouble(C)||maximum(C)>1)
    C = double(C)/255;
end
end