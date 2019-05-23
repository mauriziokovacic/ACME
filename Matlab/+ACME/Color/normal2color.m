function [C] = normal2color(N)
% NORMAL2COLOR  Converts a given set of normals into a color vector
%   C = NORMAL2COLOR(N) returns a nx3 color vector from a nx3 matrix N.
C = normr(N);
C = (C + 1).*0.5;
end