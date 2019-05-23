function [C] = Magenta()
% MAGENTA  Returns the magenta color as the complementary of Green()
%   C = MAGENTA() returns a 1x3 vector.
%
% See also GREEN.
C = 1-Green();
end