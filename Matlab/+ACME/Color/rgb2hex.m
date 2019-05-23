function [H] = rgb2hex(C)
% RGB2HEX  Converts a color vector in RGB format into hex format.
%   H = RGB2HEX(C) returns a 1x3 vector.
H = color2hex(C);
end