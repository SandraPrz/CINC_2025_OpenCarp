%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      function VTKstruct = vtk2structReader (fullPath)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   This function reads all the information contained in a .VTK file in
%   ASCII format and stores it into a matlab structure called 'VTKstruct'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs  ------------------------------------------------------------- %
%       fullPath  ->  Full path of the .VTK file to be read
%                       IMPORTANT. If 'fullPath' is not specified as an
%                         input, the user will be asked for the location
%                         of the .VTK file to be read by means of a dialog
%                         box
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INFO  --------------------------------------------------------------- %
%
%  This function is for 3D surfaces
%
%  Supported .vtk DATASETs:
%   -   POLYDATA
%   -   UNSTRUCTURED_GRID
%
%  Supported POINT_DATA:
%   -   SCALARS
%   -   NORMALS
%   -   VECTORS
%
%  Supported CELL_DATA:
%   -   SCALARS
%   -   NORMALS
%   -   VECTORS
%
%  Others:
%   -   FIELD
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP ---------------------------------------------------------- %
%                       Author:     Alejandro Daniel López Pérez
%                Creation date:     ???
%       Last modification date:     08/05/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function VTKstruct = vtk2structReader (fullPath)

%% INPUT CHECK  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1.- Was the fullPath of the .VTK file specified as an input ??????????
if nargin < 1  % if NOT
    % Ask the user for the .VTK file to be read
    [file, path] = uigetfile('*.vtk', 'Choose a .VTK file');
    % Check whether a file was properly chose
    if file ~= 0  % if so
        % Create the 'fullPath'
        fullPath = fullfile (path, file);
    else  % if NOT
        errordlg('No file was selected', ...
            'Error using :: vtk2structReader')
        VTKstruct = [];
        return
    end
end  % if so, continue

%%% 2.- Is it a .VTK file ???????????????????????????????????????????????? 
% Check whether the provided 'fullPath' corresponds to a .VTK file
if ~strcmp ('.vtk', fullPath(end-3:end))  % If not
    errordlg('The provided path does not correspond to a .VTK file.', ...
        'Error using :: vtk2structReader')
    VTKstruct = [];
    return
end  % If so, continue


%% OPEN THE .VTK FILE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check whether the specified .VTK file really exists
if exist (fullPath, 'file')  % if so
    % Open the .VTK file
    fid = fopen (fullPath);
else  % if NOT
    errordlg('The specified .VTK file does not exist', ...
        'Error using :: vtk2structReader')
    VTKstruct = [];
    return
end


%% IS A 'ASCII' .VTK FILE ???  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Look for the ASCII label in the VTK file
while 1  % Start going through the .VTK file
    
    % Read the next line in the VTK file
    tline = fgetl (fid);
    
    % Check if the ASCII label has been reached
    if strfind (tline, 'ASCII')  % If so        
        break  % Leave the 'while' loop and continue
    end
    
    % Check if the 'eof' (end-of-file) has been reached
    if feof(fid)
        fclose (fid);  % Close the file
        errordlg (['This .VTK file is not in ASCII format:  ' ...
            fullPath], 'ERROR reading .VTK file')
        VTKstruct = [];
        return         % Leave the function
    end  
end


%% CREATE THE 'VTKstruct'  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VTKstruct = struct ('Dataset', [], ...
    'NumPoints', [], ...
    'Points', [], ...
    'NumCells', [], ...
    'Cells', [], ...
    'PointData', [], ...
    'CellData', [], ...
    'Field', [], ...
    'CellTypes', [], ...
    'FilePath', []);

% Save file path in 'VTKstruct'
VTKstruct.FilePath = fullPath;


%% LOOK FOR 'DATASET' SECTION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Look for the DATASET section in the .VTK file
while 1   % Keep going through the .VTK file

    % Read the next line in the VTK file
    tline = fgetl (fid);
    
    % Check if the 'eof' (end-of-file) has been reached
    if feof(fid) %#ok<*UNRCH>
        errordlg (['No DATASET section was found in this file:  ' ...
            fullPath], 'ERROR reading .VTK file')
        VTKstruct = [];
        fclose (fid);  % Close the file
        return         % Leave the function
    end
        
    % Check if the DATASET section has been reached
    if strfind (tline, 'DATASET')  % If so
        
        % Get the 'Dataset' of this .VTK file
        info = textscan(tline, '%*s %s');
        
        % Store the .vtk DATASET type in 'VTKstruct'
        VTKstruct.Dataset = char(info{1});            %%%  VTKstruct.Dataset  =  { 'POLYDATA', 'UNSTRUCTURED_GRID' }
        
        % Check the .vtk dataset type
        switch char(info{1})
            case {'POLYDATA', 'UNSTRUCTURED_GRID'}
                break   % Leave the 'while' loop and continue
            otherwise
                % Error message
                errordlg (sprintf('Only the .vtk DATASET listed below are supported: \n\n    -  POLYDATA \n    -  UNSTRUCTURED_GRID'), ...
                    'ERROR reading .VTK file')
                fclose (fid);  % Close the file
                return   % Leave the function
        end
    end
end


%% LOOK FOR 'POINTS' SECTION:  POINT COORDINATES  %%%%%%%%%%%%%%%%%%%%%%%%%
% Look for POINTS section in the .VTK file
while 1   % Keep going through the .VTK file
    
    % Read the next line in the .VTK file
    tline = fgetl (fid);
    
    % Check if the 'eof' (end-of-file) has been reached
    if feof(fid) %#ok<*UNRCH>
        errordlg (['No POINTS section was found in this file:  ' ...
            fullPath], 'ERROR reading .VTK file')
        fclose (fid);  % Close the file
        return         % Leave the function
    end

    % Check if the POINTS section has been reached
    if strfind (tline, 'POINTS')  % If so
        % And get the 'NUMBER OF POINTS' from this line
        VTKstruct.NumPoints = cell2mat(textscan(tline,'%*s %d %*s'));       %%%  VTKstruct.NumPoints
        % Extract the coordinates of the points from the VTK file
        VTKstruct.Points = fscanf (fid, '%f %f %f', ...                     %%%  VTKstruct.Points  ->  Nx3 matrix (X,Y,Z coordinates of each point)
            [3, VTKstruct.NumPoints])';        
        break    % Leave the 'while' loop and continue
    end
end


%% LOOK FOR ANY OTHER SECTION IN THE .VTK FILE  %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the next line in the VTK file
tline = fgetl (fid);

% Go through the rest of the .VTK file looking for any other VTK section
while ~feof(fid)  % Continue while the 'end-of-file' is not reached

    % Check whether the current line is empty    
    if ~isempty (tline)  % If not 
        % Get the information contained in this line
        info = textscan (tline, '%s');
    
        % Check whether the line consists of blank spaces
        if ~size(info{1},1) == 0   % 

            % Check the first word in the this line
            switch info{1}{1}  % The first word of the line is in 'info{1}{1}'

%%%%%%%%%%%%%%% Look for the CONNECTIVITY LIST defining the triangles
                %   - POLYGONS section in the case of :: VTKstruct.Dataset = 'POLYDATA'
                %   - CELLS section in the case of :: VTKstruct.Dataset = 'UNSTRUCTURED_GRID'
                case {'POLYGONS', 'CELLS', 'LINES'}
                    % Get the number on triangles from the current line
                    info = cell2mat (textscan(tline,'%*s %d %d'));
                    VTKstruct.NumCells = info (1);  %%%  VTKstruct.NumCells

                    % Check whether all defined polygons/cells are triangles
                    if 4*info(1) == info(2)   % If so
                        
                        % Get the connectivity list for TRIANGLES
                        VTKstruct.Cells = fscanf (fid, '%*d %d %d %d', ...
                            [3, info(1)])';
                        
                    else   % Otherwise
                        
                        % Define 'VTKstruct.Cells' as a 'cell array'
                        VTKstruct.Cells = cell (info(1), 2);
                        % Save the current position in the file
                        position = ftell (fid);
                        % Calculate the supposed number of data to be read from each line
                        numDataPerLine = info(2) / info(1);
                        % Check whether this number is an integer
                        if isinteger (numDataPerLine)  % if so
                            % Then assume all CELLS are of the same type (same number of nodes/points)
                            % and read the data under this assumption
                            pattern = repmat ('%d ', 1, numDataPerLine);
                            data = fscanf (fid, pattern, [numDataPerLine, info(1)])';
                            % Check whether the previous assumption is true
                            if logical(sum(data(:,1) ~= numDataPerLine-1))  % if it is NOT true
                                % Go back through the file to the position corresponding to the
                                % beginning of the CELLS section
                                fseek (fid, position, 'bof');
                                % And go through the CELLS section line by line in order to read again the data
                                for i = 1:info(1)
                                    % Read the next line in the VTK file
                                    tline = fgetl (fid);
                                    % Get the information about the cells from the current line
                                    vertex = cell2mat (textscan (tline, '%d'));
                                    % Store the number of vertices of the current cell in the first column
                                    VTKstruct.Cells{i,1} = vertex(1);
                                    % Store the list of vertices that form the current cell in the second column
                                    VTKstruct.Cells{i,2} = vertex(2:end);
                                end
                            else  % if the assumption is true
                                % Save the read data
                                VTKstruct.Cells(:,1) = num2cell (data(:,1), 2);
                                VTKstruct.Cells(:,2) = (num2cell (data(:,2:end)', 1))';
                            end
                        end
                    end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
%%%%%%%%%%%%%%% Look for CELL_TYPES section (only for UNSTRUCTURED_GRID)
                case 'CELL_TYPES'
                    % Store the 'CELL_TYPES' data in 'VTKstruct.CellTypes'
                    VTKstruct.CellTypes = fscanf (fid, '%d', ...
                        [1, VTKstruct.NumCells])';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Look for the beginning of POINT_DATA section            
                case 'POINT_DATA'
                    % Reset the 'counter'
                    counter = 0;
                    % Set 'PointData' as the 'field' to store the data in the 'VTKstruct'
                    structField = 'PointData';
                    % Set 'VTKstruct.NumPoints' as the number of data to be read in this section
                    numData = VTKstruct.NumPoints;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Look for the beginning of CELL_DATA section
                case 'CELL_DATA'
                    % Reset the 'counter'
                    counter = 0;
                    % Set 'CellData' as the 'field' to store the data in the 'VTKstruct'
                    structField = 'CellData';
                    % Set 'VTKstruct.NumCells' as the numbar of data to be read in this section
                    numData = VTKstruct.NumCells;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    

%%%%%%%%%%%%%%% Look for SCALARS data, both for POINT_DATA and CELL_DATA
                case 'SCALARS'
                    % Update the 'counter'
                    counter = counter + 1;
                    % Get the information contained in the current line
                    info = textscan (tline, '%*s %s %s %d');
                    
                    % Store the metadata in 'VTKstruct'
                    VTKstruct = setfield (VTKstruct, structField, {counter}, ...    % SCALARS
                        'Type', 'SCALARS');
                    VTKstruct = setfield (VTKstruct, structField, {counter}, ...    % Name of data
                        'Name', char(info{1}));
                    VTKstruct = setfield (VTKstruct, structField, {counter}, ...    % Data format: 'float', 'int', ...
                        'Format', char(info{2}));

                    % Move forward one line to skip the LOOKUP_TABLE line
                    fgetl (fid);

                    % Check the 'dataType' in order to establish the pattern
                    switch char(info{2})
                        case {'float', 'double'}  % If float
                            pattern = '%f';
                        otherwise  % Otherwise, assume it is an integer
                            pattern = '%d';
                    end

                    % Check 'numComp' info (number of components) 
                    if isempty (info{3})  % If it is missing
                        numComp = 1;   % Assume 1 as default value (VTK standard)
                    else  % Otherwise
                        numComp = info{3};   % Take the specified value
                    end

                    % Get the SCALARS data
                    VTKstruct = setfield (VTKstruct, structField, {counter}, ...
                        'Data', fscanf(fid, pattern, [numComp, numData])');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        

%%%%%%%%%%%%%%% Look for NORMALS or VECTORS data, both for POINT_DATA and CELL_DATA            
                case {'NORMALS', 'VECTORS'}
                    % Update the 'counter'
                    counter = counter + 1;
                    % Get the information contained in the current line
                    info = textscan (tline, '%s %s %s');
                    
                    % Store the metadata in 'VTKstruct'
                    VTKstruct = setfield (VTKstruct, structField, ...       % NORMALS or VECTORS
                        {counter}, 'Type', char(info{1}));          
                    VTKstruct = setfield (VTKstruct, structField, ...       % Name of data
                        {counter}, 'Name', char(info{2}));
                    VTKstruct = setfield (VTKstruct, structField, ...       % Data format: 'float', 'int', ...
                        {counter}, 'Format', char(info{3}));

                    % Check the dataType in order to establish the pattern
                    switch char(info{3})
                        case {'float', 'double'}  % If float
                            pattern = '%f %f %f';
                        otherwise  % Otherwise, assume it is as an integer
                            pattern = '%d %d %d';
                    end
                    
                    % Get the NORMALS or VECTORS data (x,y,z components)
                    VTKstruct = setfield (VTKstruct, structField, {counter}, ...
                                'Data', fscanf(fid, pattern, [3, numData])');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Look for FIELD section
                case 'FIELD'
                    % Reset the 'counter'
                    counter = length (VTKstruct.Field);
                    % Get the number of FIELDs
                    numField = counter + cell2mat (textscan (tline, '%*s %*s %d'));
                    % Read the next line in the VTK file
                    tline = fgetl (fid);
                    
                    % Go through the FIELD section
                    while counter < numField
                        
                        % Check whether the current line is empty    
                        if ~isempty (tline)  % If not 
                            % Get the information contained in this line
                            info = textscan (tline, '%s');
    
                            % Check whether the line consists of blank spaces
                            if ~size(info{1},1) == 0   % 
                        
                                % Extract the information contained in the current line
                                info = textscan (tline, '%s %d %d %s');

                                % Update the 'counter'
                                counter = counter + 1;
                                % Store the metadata in the 'VTKstruct'
                                VTKstruct = setfield (VTKstruct, 'Field', {counter}, ...    % Name of FIELD data
                                    'Name', char(info{1}));
                                VTKstruct = setfield (VTKstruct, 'Field', {counter}, ...    % Data format: 'float', 'int', ...
                                    'Format', char(info{4}));

                                % Check the data format in order to establish the pattern to be read
                                switch char(info{4})
                                    case {'float', 'double'}
                                        pattern = '%f';
                                    otherwise
                                        pattern = '%d';
                                end

                                % Get the FIELD data
                                VTKstruct = setfield (VTKstruct, 'Field', {counter}, ...    % Data format: 'float', 'int', ...
                                    'Data', fscanf(fid, pattern, [info{2}, info{3}])');
                            end
                        end

                        % Read the next line in the VTK file
                        tline = fgetl (fid);
                    end

                    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
    end
    
    % Read the next line in the VTK file
    tline = fgetl (fid);
    
end


%% FINISH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Close the .VTK file
fclose (fid);