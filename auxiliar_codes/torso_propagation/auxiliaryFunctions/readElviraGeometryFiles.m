%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%     elviraModel = readElviraGeometryFiles (elviraFolder)      %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   'readElviraGeometryFiles' reads all files defining the domain
%   (geometry) of an Elvira heart model in order to extract all data from
%   them and returns all information into a Matlab structure (output
%   'elviraModel')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ------------------------------------------------------------- %%
%      elviraFolder  ->  String specifying the full path of a directory
%                        that contains all geometry Elvira files of a
%                        heart model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUTs ------------------------------------------------------------ %%
%      elviraModel      ->  Matlab structure containing the following fields 
%           .NumNodes          ->  Number of nodes of heart model
%           .Coordinates       ->  x,y,z coordinates of all nodes of heart model
%           .NumElements       ->  Number of cells/elements of heart model
%           .ConnectivityList  ->  Connectivity list of all elements of heart model
%           .FibreOrientation  ->  x,y,z components of a unit vector defining the fibre orientation for each element of heart model
%           .Material          ->  Label defining which material each element of heart model belongs to
%           .LongCV            ->  Longitudinal conductivity for each material existing in heart model
%           .AnisotropyRatio   ->  Anisotropy ratio (trans. CV/long. CV) for each material existing in heart model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP -------------------------------------------------------- %%%
%                  Author:     Alejandro D. Lopez Perez
%           Creation date:     ??/??/2016
%       Last modification:     14/06/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function elviraModel = readElviraGeometryFiles (elviraFolder)

% Look for ELVIRA geometry files in specified folder
datFiles = dir (fullfile (elviraFolder, '*.dat'));
% Go through the files contained in the selected folder looking for the
% Elvira's geometry files:
% {'MATERIALS_', 'PROP_ELEM_', 'NODES_', 'ELEMENTS'}
elviraFiles = struct ('Nodes', [], 'Elements', [], 'PropElem', [], 'Materials', []);
for i = 1:length(datFiles)
    if strncmp (datFiles(i).name, 'NODES', 5)
        elviraFiles.Nodes = fullfile (elviraFolder, datFiles(i).name);
    elseif strncmp (datFiles(i).name, 'ELEMENTS', 8)
        elviraFiles.Elements = fullfile (elviraFolder, datFiles(i).name);
    elseif strncmp (datFiles(i).name, 'PROP_ELEM', 9)
        elviraFiles.PropElem = fullfile (elviraFolder, datFiles(i).name);
    elseif strncmp (datFiles(i).name, 'MATERIALS', 9)
        elviraFiles.Materials = fullfile (elviraFolder, datFiles(i).name);
    end
end

% Create 'elviraModel' structure
elviraModel = struct ('NumNodes', [], 'Coordinates', [], 'NumElements', [], ...
                      'ConnectivityList', [], 'FibreOrientation', [], 'Material', [], ...
                      'LongCV', [], 'AnisotropyRatio', []);

% Reading nodes and elements from Elvira data files
%%% NODES file
tic
fprintf ('        *  Reading NODES file');
fp = fopen (elviraFiles.Nodes, 'r');                % Open file
data = textscan (fgetl(fp), '%d %*d %d');           % Read frist line
fgetl(fp);                                          % Skip 1 line
elviraModel.NumNodes = double (data{1});            % Get number of nodes
problemDim = double (data{2});                      % Get problem dimension (usually 3D)
pattern = ['%*d %*d', repmat(' %f',1,problemDim)];  % Create pattern to get coordinates
% Get nodes coordinates (x,y,z coordinates in the case of 3D problem)
elviraModel.Coordinates = cell2mat (textscan (fp, pattern, elviraModel.NumNodes, 'commentStyle', '!'));
fclose(fp);     % Close file
fprintf (':   %s \n', executionTime(toc));

%%% ELEMENTS file
tic
fprintf ('        *  Reading ELEMENTS file');
fp = fopen (elviraFiles.Elements, 'r');                     % Open file
fullNumElem = cell2mat (textscan (fgetl(fp), '%d', 1));     % Get full number of elements from first line
data = textscan (fgetl(fp), '%d %s');                       % Get data from 2nd line
elviraModel.NumElements = data{1};                          % Number of element of the first type
nodesPerElem = str2double (data{2}{1}(end));                % Nmber of nodes per element
pattern = ['%*d %*d', repmat(' %d',[1,nodesPerElem])];      % Create pattern to retrieve the connectivity list
% Get connectivity list
elviraModel.ConnectivityList = double (cell2mat (textscan (fp, pattern, elviraModel.NumElements, 'commentStyle', '!')));
fclose (fp);                                     % Close file
fprintf (':   %s \n', executionTime(toc));
% Check whether there is any other type of element
if fullNumElem ~= elviraModel.NumElements
    warning (' -> Elvira ELEMENT file <-  :::  There are more than one type of element');
end

%%% PROP_ELEM file -> Fibre orientation
tic
fprintf ('        *  Reading PROP_ELEM file {Fibre Orientation}');
fp = fopen (elviraFiles.PropElem, 'r');         % Open file
fgetl (fp);                                     % Skip first line
pattern = ['%*d %d %*d %*d %*d', repmat(' %f',1,problemDim)];   % Create pattern to get data from PROP_ELEM file
data = textscan (fp, pattern, elviraModel.NumElements, 'commentStyle', '!');    % Get data
fclose (fp);        % Close file
elviraModel.Material = double (data{1});                            % Take material identifiers from extracted data
elviraModel.FibreOrientation = double (cell2mat (data(2:end)));     % Take fibre orientation vector from extracted data
fprintf (':   %s \n', executionTime(toc));

%%% MATERIALS file
tic
fprintf ('        *  Reading MATERIALS file');
fp = fopen (elviraFiles.Materials, 'r');                    % Open file
numMaterials = cell2mat (textscan (fgetl(fp), '%d', 1));    % Get number of materials defined
pattern = '%*d %*d %*d %*d %*d %f %f %*d';                  % Create pattern to get data related to defined materials
data = textscan (fp, pattern, numMaterials, 'commentStyle', '!');   % Get data from MATERIALS file
fclose (fp);                                                % Close file
elviraModel.LongCV = data{1};                               % Take longitudinal conductivity from extracted data
elviraModel.AnisotropyRatio = data{2};                      % Take anisotropy ratio from extracted data
fprintf (':   %s \n', executionTime(toc));