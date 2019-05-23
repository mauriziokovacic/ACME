function [C] = fetch_color(CData,FData)
% FETCH_COLOR  Interpolates the color of a given linear texture, using a
% given set of values
%   C = FETCH_COLOR(CData,FData) interpolates CData using FData. CData is a
%   nx3 color vector, while FData is a mx1 scalar vector
FData = normalize(FData);
C     = interp1(linspace(0,1,row(CData)),CData,FData);
end