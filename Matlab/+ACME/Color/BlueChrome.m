function [CData] = BlueChrome(CRes)
% BLUECHROME  Returns the blue chrome color vector.
%   C = BLUECHROME(CRes) returns a CRes x 3 vector.
CData = Chrome([0 1 1; 0 0 1; 0.5 0 1; 0.5 0 1],CRes);
end