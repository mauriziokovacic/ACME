function [C] = scatter_color( NColors, ID, Saturation, Value )
% SCATTER_COLOR  Assigns a different color to a given ID, choosing from a
% given number of colors
%   C = SCATTER_COLOR(NColors,ID) returns a color per ID, choosing from a
%   set of NColors
%   C = SCATTER_COLOR(NColors,ID,Saturation) assigns a Saturation level to
%   all the returned colors.
%   C = SCATTER_COLOR(NColors,ID,Saturation,Value) assigns a Value level to
%   all the returned colors.
if( nargin < 4 )
    Value = 0.85;
end
if( nargin < 3 )
    Saturation = 0.5;
end
if( row(ID)==1 )
    ID = ID';
end
Saturation = repmat(Saturation,numel(ID),1);
Value      = repmat(Value,numel(ID),1);

NColors = NColors+1;

M = repmat(NColors,numel(ID),1);
R = repmat(NColors,numel(ID),1);
B = zeros(numel(ID),1);

k = 2.^(0:32)';
k = k(k<NColors);
for n = 1 : numel(k)
    i = find(bsxfun(@(a,b) a>=b,bitshift(ID,1),M));
    if( ~isempty(i) )
        R(B(i)==0) = k(n);
        B(i) = B(i)+k(n);
        ID(i) = ID(i) - bitshift(M(i)+1,-1);
        M(i) = bitshift(M(i),-1);
    end
    j = setdiff((1:row(ID))',i);
    if( ~isempty(j) )
        M(j) = bitshift(M(j)+1,-1);
    end
    
    i = find(R>(NColors-B));
    if( ~isempty(i) )
        R(i) = NColors-B(i);
    end
end
C = hsv2rgb([B/NColors, Saturation, Value]);
end