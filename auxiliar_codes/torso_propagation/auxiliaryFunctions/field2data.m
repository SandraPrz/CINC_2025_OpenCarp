%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         function outVTKstruct = field2data (inVTKstruct)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This function transforms all FIELDs contained in 'inVTKstruct' into
%  POINT_DATA or CELL_DATA fields in 'outVTKstruct'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function outVTKstruct = field2data (inVTKstruct)

% Get the number of existing FIELDs in 'inVTKstruct'
numFields = length (inVTKstruct.Field);
% Check whether actually exists any FIELD in 'inVTKstruct'
if numFields == 0  % If not
    warndlg ('This VTKstruct does NOT contain any FIELD', 'Warning');
    return   % Leave the function
end  % If so, continue

% Copy 'inVTKstruct' into 'outVTKstruct'
outVTKstruct = inVTKstruct;
% Remove all FIELDs from 'outVTKstruct'
outVTKstruct.Field = [];

% Get the number of existing POINT_DATA in 'inVTKstruct' and initialise an index for new POINT_DATA in 'outVTKstruct'
indPointData = 1 + length (inVTKstruct.PointData);
% Get the number of existing CELL_DATA in 'inVTKstruct'
indCellData = 1 + length (inVTKstruct.CellData);

% Convert all FIELDs in 'inVTKstruct' into POINT_DATA/CELL_DATA in 'outVTKstruct' 
for i = 1:numFields
    
    % Check whether the current FIELD corresponds to POINTs or CELLs
    switch size(inVTKstruct.Field(i).Data, 1)
        
        % if POINTS
        case inVTKstruct.NumPoints
            % Create a new POINT_DATA
            % As FIELD has NO 'type', all new FIELDS are set as 'SCALARS'
            outVTKstruct.PointData(indPointData).Type = 'SCALARS';
            % 'Name' of new POINT_DATA
            outVTKstruct.PointData(indPointData).Name = ...
                inVTKstruct.Field(i).Name;
            % 'Format' of new POINT_DATA
            outVTKstruct.PointData(indPointData).Format = ...
                inVTKstruct.Field(i).Format;
            % 'Data' of new POINT_DATA
            outVTKstruct.PointData(indPointData).Data = ...
                inVTKstruct.Field(i).Data;
            % Update the index for new POINT_DATA in 'outVTKstruct'
            indPointData = indPointData + 1;
    
        % if CELLS
        case inVTKstruct.NumCells
            % Create a new CELL_DATA
            % As FIELD has NO 'type', all new FIELDS are set as 'SCALARS'
            outVTKstruct.CellData(indCellData).Type = 'SCALARS';
            % 'Name' of new CELL_DATA
            outVTKstruct.CellData(indCellData).Name = ...
                inVTKstruct.Field(i).Name;
            % 'Format' of new CELL_DATA
            outVTKstruct.CellData(indCellData).Format = ...
                inVTKstruct.Field(i).Format;
            % 'Data' of new CELL_DATA
            outVTKstruct.CellData(indCellData).Data = ...
                inVTKstruct.Field(i).Data;
            % Update the index for new CELL_DATA in 'outVTKstruct'
            indCellData = indCellData + 1;
            
        % if not POINTS nor CELLS
        otherwise
            % Show a warning message
            warning (sprinft('field2data  :::  The data length of the FIELD called ''%s'' does not correspond neither to the number of POINTs nor to the number of CELLs defined',inVTKstruct.Field(i).Name));
    end

end