function [C] = Cyan()
% CYAN  Returns the cyan color as the complementary of Red.
%   C = CYAN() returns a 1x3 vector.
%
% See also RED
C = 1-Red();%[0 1 1];
end