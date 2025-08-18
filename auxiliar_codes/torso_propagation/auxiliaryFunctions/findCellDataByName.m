%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%  [index, data] = findCellDataByName (VTKstruct, fieldName)  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   'findCellDataByName' returns in the output 'index' the index of the
%   CELL_DATA field of 'VTKstruct' whose name matches the string given in
%   'fieldName'. If 'data' is also requested as an output, the data
%   associated to the field will be returned in it.
%  ---------------------------------------------------------------------  %
%  IMPORTANT. If no matching is found, 'index' and 'data' will be returned
%             as empty matrices.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Author:     Alejandro Daniel López Pérez
%            Creation date:     ????
%   Last modification date:     11/05/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [index, data] = findCellDataByName (VTKstruct, fieldName)

% Initialise 'index' as an empty matrix
index = [];

% Check whether actually exists any field in the CELL_DATA section of 'VTKstruct'
numCellDataFields = length (VTKstruct.CellData);
if numCellDataFields == 0 % if empty
    return
end % otherwise, continue

% Go through the CELL_DATA section of 'VTKstruct'
for i = 1:numCellDataFields
    % Check whether the 'Name' of the current CELL_DATA field matches the sought one
    if strcmp (VTKstruct.CellData(i).Name, fieldName)
        % if so
        index = i;  % save the index
        break       % and leave the for-loop
    end
end

% If it is requested as an output, get the data from the searched CellData field
if nargout > 1  % if so
    % Check whether the CellData field was found
    if isempty (index)  % if NOT
        % Return an empty matrix
        
    else  % if so
        % Get the data
        data = VTKstruct.CellData(i).Data;
    end
end