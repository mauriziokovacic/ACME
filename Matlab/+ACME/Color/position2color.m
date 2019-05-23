function [C] = position2color(P,Min,Max)
% POSITION2COLOR  Converts a 3-dimensional coordinate into a color.
%   C = POSITION2COLOR(P) returns a nx3 vector, with C(i,:) representing
%   the color P(i,:).
%   C = POSITION2COLOR(P,Min) returns a nx3 vector, where the color black
%   represents the coordinate Min
%   C = POSITION2COLOR(P,Min,Max) returns a nx3 vector, where the color
%   black represents the coordinate Min, while white represents Max
if( nargin < 3 )
    Max = max(P);
end
if( nargin < 2 )
    Min = min(P);
end
C = normalize(clamp3(P,Min,Max),Min,Max);
end