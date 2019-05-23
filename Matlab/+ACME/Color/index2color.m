function [C] = index2color(ID)
% INDEX2COLOR  Converts a index into a color with values in [0-255]
%   C = INDEX2COLOR(ID) returns a nx3 vector, with C(i,:) representing the
%   color coding of ID(i).
ID = uint32(ID(:));
C = reshape(typecast(ID,'uint8'),4,row(ID))';
C = [C(:,1) C(:,2) C(:,3)];
end