%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    [torsoElements, torsoNodes] = readTorsoModelFile (fileName, tissueFile, myocardLabel, organLabel)   %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  'readTorsoModelFile' reads the VTK file containing a labeled torso
%  model and creates the following .mat files:
%       *   NODES file     ->  Nx3 matrix containing the x,y,z coordinates
%                              of all points/nodes of torso model, where N
%                              is the number of nodes in torso model
%       *   ELEMENTS file  ->  MxK matrix where each row contains the
%                              connectivity list of the corresponding
%                              cell/element of torso model and also a
%                              'tissueID' label, the conductivity of that
%                              tissue ('sigma') and the ratio between
%                              transversal and longitudinal conductivity
%                              ('a'). M is the number of cells/elements of
%                              torso model and K will depend on the element
%                              type of the volume mesh (number of
%                              points/nodes per cell/element)
%       *   NODES MYOCARDIUM file  ->  Array containing the Matlab/Elvira
%                                      indices (starting at '1') of all
%                                      points/nodes of torso model
%                                      belonging to cells/elements labeled
%                                      as myocardium
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs -------------------------------------------------------------- %
%     torsoFilePath  ->  Full path of the VTK file containing the TORSO model
%                     It does not matter which type of element the torso
%                     model is built with {TETRAHEDRA,HEXAHEDRA,VOXEL ...}
%     tissueFile  ->  String containing the full path of a text file
%                         (.txt, .dat, ...) containing two data columns.
%                           * 1st column contains the value of t he label
%                             that defines the different organs/tissues.
%                           * 2nd column contains the value of the conduction
%                             velocity (CV) of each tissue/organ present in
%                             torso model
%     destFolder  ->  String containing the path of the folder where new
%                     .mat files must be saved
%     myocardLabel  ->  Integer value corresponding to the value of label
%                       of torso model defining 'myocardium' cells/elements
%     organLabel  ->  { OPTIONAL } String containing the name of the CELL_DATA
%                     field of the torso model corresponding to the label that
%                     defines the different organs and tissues
%                       IMPORTANT  ->  It is case sensitive !!!!!!!!!!
%                       IMPORTANT  ->  If it is NOT specified as a input or
%                            is specified as a emtpy matrix, it will be assumed
%                            that the first CELL_DATA field contains the label
%                            defining different tissues and/or organs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LIST OF FUNCTIONS CALLED FROM 'my_A_PrepareTorsoElements':
%       *  from 'my_VTK_ToolBox':
%            -  vtk2structReader
%            -  field2data
%            -  getConnectivityList
%            -  findCellDataByName
%       *  Others:
%            -  executionTime
%            -  setNewFileName
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP ---------------------------------------------------------- %
%                  Author:     Jose Felix Rodriguez ?????
%           Creation date:     ??/??/????
%       Last modification:     09/03/2017  by  Alejandro D. Lopez Perez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % function [nodesCoord, elemMatrix, cellsMyocard, nodesMyocard] = readTorsoModelFile (filePath, tissueFile, destFolder, myocardLabel, organLabel)
function [torsoElements, torsoNodes] = readTorsoModelFile (torsoFilePath, tissueFile, myocardLabel, organLabel)

% Create output structures
torsoElements = struct ('NumElements', [], 'ConnectivityList', [], 'LongCV', [], ...
                        'AnisotropyRatio', [], 'OrganID', [], 'FibreOrientation', [], ...
                        'MyoID', [], 'NumMyoElements', [], 'MyoElements', []);
%%%   torsoElements.NumElements       ->  Number of elements (tetrahedra) in torso model 
%%%                .ConnectivityList  ->  Connectivity list of elements (tetrahedra) of torso model
%%%                .LongCV            ->  Longitudinal conductivity for each element of torso model
%%%                .AnisotropyRatio   ->  Anisotropy ratio (transveral CV/longitudinal CV) for each element of torso model
%%%                .OrganID           ->  Label defining to which organ/tissue belongs each element of torso model
%%%                .FibreOrientation  ->  x,y,z coordinates of the unit vector defining the fibre orientation for each element of torso model
%%%                .MyoID             ->  Value of 'OrganID' label corresponding to 'myocardium' (ventricles and/or atria)
%%%                .NumMyoElements    ->  Number of elements of torso model labeled as 'myocardium'
%%%                .MyoElements       ->  Boolean array indicating those elements of torso model labeled as 'myocardium' 
torsoNodes = struct ('NumNodes', [], 'Coordinates', [], 'NumMyoNodes', [], 'MyoNodes', []);
%%%   torsoNodes.NumNodes     ->  Number of nodes in torso model
%%%             .Coordinates  ->  x,y,z coordinates of all nodes of torso model 
%%%             .NumMyoNodes  ->  Number of nodes of torso model belonging to elements labeled as 'myocardium' 
%%%             .MyoNodes     ->  Boolean array indiciating those nodes of torso model that belong to elements labeled as 'myocardium' 


%%%%%%%%%%%%%%%
%%% Read VTK file containing the torso model (tetrahedral volume mesh)
fprintf ('    ->  Reading VTK file containing the TORSO model ...\n')
tic
torsoModel = vtk2structReader (torsoFilePath);
% Transform FIELDs into POINT_DATA or CELL_DATA fields
if ~isempty (torsoModel.Field)
    torsoModel = field2data (torsoModel);
end
fprintf ('        *  Elapsed time for reading TORSO model:  %s\n', executionTime(toc));


%%%%%%%%%%%%%%%%
%%% Create TORSO NODES file
fprintf ('    ->  Getting TORSO NODES file ... \n')
tic
% Extract x,y,z coordinates af all points/nodes of 'torsoModel'
torsoNodes.Coordinates = torsoModel.Points;
torsoNodes.NumNodes = torsoModel.NumPoints;


%%%%%%%%%%%%%%%%
%%% Create TORSO ELEMENTS file
fprintf ('    ->  Creating TORSO ELEMENTS matrix ...\n')
tic
% Get the connectiviy list
fprintf ('        *  Getting the connectivity list of TORSO model:  ');
% Save the connectivity list of the torso model in 'elemMatrix'
torsoElements.ConnectivityList = getConnectivityList (torsoModel) + 1;     %%%  Add '1' because VTK indices start at '0' while Matlab and Elvira indices start at '1'
% Number of elements in torso model
torsoElements.NumElements = torsoModel.NumCells;
% % % % % Anisotropy ratio
% % % % torsoElements.AnisotropyRatio = ones (torsoModel.NumCells, 1);    %%%  Anisotropy ratio (transveral CV / longitudinal CV)
fprintf ('%s\n', executionTime(toc));                             %%%     ->  All organs/tissue show isotropic propagation except for the 'myocardium'

% Define conductivites for different organs and tissues
tic
fprintf ('        *  Defining conductivities for different organs and tissues in TORSO model:  ');
% Check whether 'organLabel' was passed as en input
if nargin > 3  % if so
    % And it is not an empty matrix
    if ~isempty(organLabel)  % if NOT
        % Look for the CELL_DATA field of 'torsoModel' containing the 'OrganID' or 'TissueID' label
        [~, organID] = findCellDataByName (torsoModel, organLabel);
    else  % if so
        % Assume there is only one CELL_DATA field that correponds to 'OrganID' or 'TissueID' label
        organID = torsoModel.CellData(2).Data;
    end
end
% Check whether the CELL_DATA field was found
if isempty (organID)  % if NOT
    error (sprintf ('\n\n  -> ERROR ::: my_A_PrepareTorsoElements  ->  ''%s'' label was not found in TORSO MODEL. Take into account that it is CASE SENSITIVE  \n', organLabel)); %#ok<SPERR>
end  % if so, just continue

% Save 'OrganID' label in output structure
torsoElements.OrganID = organID;
% Save 'myocardLabel' in output structure
torsoElements.MyoID = myocardLabel;
% Read tissue file, containing labels and CVs for different tissues/organs defined in torso model
[tissueLabel, tissueCV] = readTissueFile (tissueFile);
% Initialise field 'LongCV' of 'torsoElements' structure
torsoElements.LongCV = zeros (torsoModel.NumCells, 1);
% Save tissue properties in 'torsoElements' taking into account the label 'organID'
for i = 1:length(tissueLabel)
    % Look for all cells/elements of 'torsoModel' labeled as a certain organ/tissue
    elemOrgan = (organID == tissueLabel(i));
    % And save in 'torosElements' value for tissue properties of that organ/tissue
    torsoElements.LongCV(elemOrgan) = tissueCV (i);
end
fprintf ('%s\n', executionTime(toc));


%%%%%%%%%%%%%%%%
%%% Create MYOCARDIUM ELEMENTS file
fprintf ('    ->  Getting MYOCARDIUM ELEMENTS of torso model:  ')
tic
% Find the cells/elements of 'torsoModel' labeled as ventricular myocardium
torsoElements.MyoElements = (organID == myocardLabel);
torsoElements.NumMyoElements = sum (torsoElements.MyoElements);
fprintf ('%s\n', executionTime(toc));


%%%%%%%%%%%%%%%%
%%% Create MYOCARDIUM NODES file
fprintf ('    ->  Getting MYOCARDIUM NODES of torso model:  ')
tic
% Take from the connectivity list of 'torsoModel' all points/nodes that form myocardial cells/elements
nodesMyocard = unique (torsoElements.ConnectivityList (torsoElements.MyoElements, :));
torsoNodes.MyoNodes = false (torsoNodes.NumNodes, 1);
torsoNodes.MyoNodes(nodesMyocard) = true;
% Number of 'myocardium' nodes in torso model
torsoNodes.NumMyoNodes = sum (torsoNodes.MyoNodes);
fprintf ('%s\n', executionTime(toc));



%%  //////////////////////////////////////////////////////////////////// %%
%%% ///  AUXILIARY FUNCTIONS  ////////////////////////////////////////// %%
%%% //////////////////////////////////////////////////////////////////// %%
% This function reads 'tissue file' which contains labels and CVs for the
% different tissues/organs defined in torso model 
function [tissueLabel, tissueCV] = readTissueFile (tissueFile)
% Initialise outputs as empty matrices
tissueLabel = [];
tissueCV = [];
% Open 'tissueFile'
fid = fopen (tissueFile);
% Read following lines
while ~feof (fid)
    % Get current line
    tLine = fgetl(fid);
    % if the line is not empty
    if ~isempty (tLine)
        % Read data from current line
        %%% The line is expected to contain data:
        %%%    -  first an integer correponding to the tissue/organ label or ID
        %%%    -  and a second value which is a float defining tissue conductivity 
        data = textscan (tLine, '%d %f');
        % Check read data
        if ~isempty (data{1})
            % Take organ/tissue ID or label
            tissueLabel (end+1, 1) = data{1}; %#ok<AGROW>
            % Take conductivity
            tissueCV (end+1, 1) = data{2}; %#ok<AGROW>
        end
    end
end
% Close 'tissueFile'
fclose (fid);