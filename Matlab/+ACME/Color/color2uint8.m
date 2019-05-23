function [C] = color2uint8(C)
% COLOR2UINT8  Converts a color vector from range [0-1] to [0-255].
%   C = COLOR2UINT8(C) returns the integer version of color vector C.
if(~isa(C,'uint8'))
    C = uint8(floor(C*255));
end
end