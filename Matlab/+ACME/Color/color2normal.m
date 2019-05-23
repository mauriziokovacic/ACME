function [N] = color2normal(C)
% COLOR2NORMAL  Converts a given color vector into normal directions.
%   N = COLOR2NORMAL(C) returns a nx3 vector.
if( ~isdouble(C) )
    C = color2double(C);
end
N = (C .* 2)-1;
end