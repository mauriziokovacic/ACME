function [varargout] = cmap(CData,CRes,isoline,CIso,varargin)
% CMAP  Assigns a color map to the current figure
%   C = CMAP(CData) assigns the given CData texture or palette to the
%   figure color map.
%   C = CMAP(CData,CRes) sets the number of colors to be used.
%   C = COLOR_RAMP(_,_,isoline) with isolines=true, encodes black isolines
%   into the color map
%   C = COLOR_RAMP(_,_,_,CIso) sets the isolines color to CIso
if( nargin < 4 )
    CIso = [0 0 0];
end
if( nargin < 3 )
    isoline = false;
end
if( nargin < 2 )
    CRes = 64;
end
if( ischar(CData) || isstring(CData) )
    if( strcmpi('implicit',CData) )
        CData   = implicit_field_color(CRes);
        isoline = false;
    else
        p = color_palette(CData);
        if(~isempty(p))
            CData = p;
        else
            CData = colormap(CData);
        end
    end
end
% C = colormap(clamp(color_ramp(CData,CRes,isoline,CIso,varargin{:}),0,1));
C = (clamp(color_ramp(CData,CRes,isoline,CIso,varargin{:}),0,1));
if(nargout==0)
    colormap(C);
else
    varargout{1} = C;
end
end