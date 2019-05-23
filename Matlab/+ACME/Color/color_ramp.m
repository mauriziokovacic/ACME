function [C] = color_ramp(CData,CRes,isoline,CIso,varargin)
% COLOR_RAMP  Creates a linear color texture from a given set of colors
%   C = COLOR_RAMP(CData) returns a 64x3 color vector created interpolating
%   the values of CData.
%   C = COLOR_RAMP(CData,CRes) returns a CResx3 color vector.
%   C = COLOR_RAMP(CData,_,isoline) with isolines=true, returns a nx3 color
%   vector with built-in black isolines
%   C = COLOR_RAMP(CData,_,isoline,CIso) sets the isolines color to CIso
if( nargin < 4 )
    CIso = [0 0 0];
end
if( nargin < 3 )
    isoline = false;
end
if( nargin < 2 )
    CRes = 64;
end
if( isoline )
    delta = 16;
    C = repelem(color_gradient(CData,20,varargin{:}),delta,1);
    i = setdiff((1:delta:row(C))',[1,row(C)]);
    C(i,:) = repmat(CIso,numel(i),1);
else
    C = color_gradient(CData,CRes,varargin{:});
end
end