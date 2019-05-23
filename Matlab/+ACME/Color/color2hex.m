function [C] = color2hex(C)
% COLOR2HEX  Converts a given color vector into hex format.
%   C = COLOR2HEX(C) returns the string representation of the input colors 
%   in hex format.
if(isdouble(C))
    C = color2uint8(C);
end
C = cell2mat(reshape(cellstr(dec2hex(C,2)),row(C),3));
end