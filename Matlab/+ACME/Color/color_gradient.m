function [C] = color_gradient(CData,CRes,varargin)
% COLOR_GRADIENT  Creates a color gradients from the given color vector.
%   C = COLOR_GRADIENT(CData,CRes) returns a CResx3 color vector derived
%   from CData
CParam = linspace(0,1,row(CData))';
CRes   = linspace(0,1,CRes)';
C      = interp1(CParam,CData,CRes,varargin{:});
end