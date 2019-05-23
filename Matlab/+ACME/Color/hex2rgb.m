function [C] = hex2rgb(H)
% HEX2RGB  Returns the rbg color from an hex string.
%   C = HEX2RGB(H) returns a nx3 vector from the first 6 elements of H. If
%   H contains the character '#', it will be automatically ignored
C = zeros(row(H),3);
if( ischar(H) || isstring(H) )
    H = cell2mat(cellfun(@(txt) strrep(txt,'#',''),cellstr(H),'UniformOutput',false));
    C = [base2dec(H(:,1:2),16),...
    	 base2dec(H(:,3:4),16),...
    	 base2dec(H(:,5:6),16)]/255;
end
end