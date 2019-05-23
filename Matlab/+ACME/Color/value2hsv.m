function [HSV] = value2hsv(value)
% VALUE2HSV  Converts a value vector into HSV colors.
%   HSV = VALUE2HSV(value) returns a nx3 color vector in HSV format.
HSV = cat(2,normalize(value,0,2/3),ones(row(value),2));
end