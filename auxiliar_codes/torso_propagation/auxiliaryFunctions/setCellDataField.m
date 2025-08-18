% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'setCellDataField' sets a new empty CELL_DATA field in an existing
% 'VTKstruct', i.e., a Matlab structure containing the information
% extracted from a .VTK file by means of the function 'vtk2structReader'
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% AUTHORSHIP --------------------------------------------------------- %
%                       Author:     Alejandro Daniel López Pérez          %
%                Creation date:     ??/??/2016                            %
%       Last modification date:     13/06/2016                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [VTKstruct, ind] = setCellDataField (VTKstruct)

% Set a new 'CELL_DATA' field into the 'VTKstruct' passed as an input
ind = length (VTKstruct.CellData) + 1;
VTKstruct.CellData(ind).Type = [];
VTKstruct.CellData(ind).Name = [];
VTKstruct.CellData(ind).Format = [];
VTKstruct.CellData(ind).Data = [];