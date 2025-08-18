%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   step1_generateFilesForTorsoPropagation (elviraFolder, torsoModelFile, tissueFile , myocardLabel, organID)   %%%
%%%%%%%%%%%%%%%%%%%$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%%%         1st STEP IN TORSO PROPAGATION PROCESS         %%%%%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  'step1_generateFilesForTorsoPropagation' prepares all data that will be
%  necessary to compute the torso propagation.
%  To do so, it reads the VTK file containing a labeled torso model and
%  also .dat files corresponding to an Elvira model (e.g., ventricles or
%  atria model) and creates the following Matlab structures:
%
%    *  TORSO ELEMENTS structure
%          torsoElements.NumElements   ->  Number of elements (tetrahedra) in torso model 
%              .ConnectivityList  ->  Connectivity list of elements (tetrahedra) of torso model 
%              .LongCV            ->  Longitudinal conductivity for each element of torso model 
%              .AnisotropyRatio   ->  Anisotropy ratio (transveral CV/longitudinal CV) for each element of torso model 
%              .OrganID           ->  Label defining to which organ/tissue belongs each element of torso model 
%              .FibreOrientation  ->  x,y,z coordinates of the unit vector defining the fibre orientation for each element of torso model 
%              .MyoID             ->  Value of 'OrganID' label corresponding to 'myocardium' (ventricles and/or atria) 
%              .NumMyoElements    ->  Number of elements of torso model labeled as 'myocardium' 
%              .MyoElements       ->  Boolean array indicating those elements of torso model labeled as 'myocardium' 
%
%    *  TORSO NODES structure
%          torsoNodes.NumNodes  ->  Number of nodes in torso model 
%                .Coordinates   ->  x,y,z coordinates of all nodes of torso model  
%                .NumMyoNodes   ->  Number of nodes of torso model belonging to elements labeled as 'myocardium'  
%                .MyoNodes      ->  Boolean array indiciating those nodes of torso model that belong to elements labeled as 'myocardium' 
%
%    *  INTERPOLATION structure
%            Matlab structure containing nodes indices and coefficients
%            to interpolate the potential values from Elvira's heart
%            model to torso model.
%
%     A new .mat file containing these structures will be saved in
%     a new folder called 'torsoPropagation', which will be created as
%     a subfolder of directory that contains 'torsoModelFile'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ------------------------------------------------------------ %%%
%       elviraFolder  ->  Path of the folder containing the Elvira's
%                         geometry files (.dat files) to be read:
%                               -   MATERIALS_ file
%                               -   PROP_ELEM_ file
%                               -   NODES_ file
%                               -   ELEMENTS_ file
%     torsoModelFile  ->  Full path of the VTK file containing the TORSO model
%                         It does not matter which type of element the torso
%                         model is built with {TETRAHEDRA,HEXAHEDRA,VOXEL ...}
%                         It is supposed to include a CELL_DATA field
%                         correspoding to an organ/tissue label
%         tissueFile  ->  String containing the full path of a text file
%                         (.txt, .dat, ...) containing two data columns.
%                           * 1st column contains the value of the label
%                             that defines the different organs/tissues.
%                           * 2nd column contains the value of the conduction
%                             velocity (CV) of each tissue/organ present in
%                             torso model
%       myocardLabel  ->  Integer value corresponding to the value of label
%                         of torso model defining 'myocardium' cells/elements
%     organID  ->  { OPTIONAL } String containing the name of the CELL_DATA 
%                  field of torso model corresponding to the label that
%                  defines the different organs or tissues
%                       IMPORTANT  ->  It is case sensitive !!!!!!!!!!!
%                       IMPORTANT  ->  If it is NOT specified as a input,
%                            it will be assumed that the first CELL_DATA
%                            field contains the label defining different
%                            tissues and/or organs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LIST OF FUNCTIONS CALLED FROM 'generateFilesForTorsoPropagation':
%       From 'my_VTK_ToolBox':
%           -  createVTKstruct
%           -  findEdges
%           -  getEdgesLength
%       Others:
%           -  readElviraGeometryFiles
%           -  readTorsoModelFile
%           -  executionTime
%           -  setNewFileName
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMPORTANT REMARKS  ------------------------------------------------ %%%
%       This function is ready to be run both in Matlab session mode and
%       also in deployed mode (compiled by 'mcc')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP -------------------------------------------------------- %%%
%                  Author:     Jose Felix Rodriguez ?????
%           Creation date:     ??/??/????
%       Last modification:     14/06/2017  by  Alejandro D. Lopez Perez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function step1_generateFilesForTorsoPropagation (elviraFolder, torsoModelFile, tissueFile, myocardLabel, organID)


elviraFolder='/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/PSM1_PSM4/PSM1_ctrl/model/'
torsoModelFile= '/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/PSM1_PSM4/PSM1_ctrl/post_S2/torso/Torso/PSM1_torso_organLabel.vtk'
tissueFile= '/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/PSM1_PSM4/PSM1_ctrl/post_S2/torso/Torso/PSM1_tissueCVs_all_normal.dat'
myocardLabel=1
organID='organLabel'
results_folder='torsoPropagation_allNormal'
addpath('/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/auxiliaryFunctions_RIGEL/')



tStart = tic;


%%  ********************************************************************* %
%%% --  PARAMETERS  ----------------------------------------------------- %
%%% ********************************************************************* %
%%% Some important parameters
% This constant ('lambda') is the ratio between the intracellular and
% extracellular conduction tensors according to the monodomain approach
lambda = 3.64;
%%% Check running mode
if isdeployed  % if running in deployed mode (comiled by 'mcc')
    % Manage numeric inputs
    myocardLabel = str2double (myocardLabel);
end   % if Matlab session, just continue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  ********************************************************************* %
%%% --  LOAD DATA  ------------------------------------------------------ %
%%% ********************************************************************* %
% Separate folder path, file name and extension
[torsoFolder, torsoFileName, ~] = fileparts (torsoModelFile);
% And create a new folder to save new .mat file
newFolder = setNewFileName (torsoFolder, results_folder);
mkdir (newFolder);
% And name for new .mat file
dataFileName = setNewFileName (newFolder, strcat (torsoFileName, '_dataForTorsoPropagation.mat'));

% Check whether it runs on a Matlab session or in deployed mode
if ~isdeployed  % if Matlab session
    % Create a log file
    diary (fullfile (newFolder, 'dataForTorsoPropagation_LOG.txt'));
end   % if deployed mode, just continue

% Read nodes and elements from Elvira data files
fprintf ('\n\n    ->  Reading ELVIRA geometry files ...\n');
elviraModel = readElviraGeometryFiles (elviraFolder);
%%% --------------------------------------------------------------------------------------------------------------------------------------- %%%
%%%   elviraModel.NumNodes          ->  Number of nodes of heart model                                                                      %%% 
%%%              .Coordinates       ->  x,y,z coordinates of all nodes of heart model                                                       %%% 
%%%              .NumElements       ->  Number of cells/elements of heart model                                                             %%% 
%%%              .ConnectivityList  ->  Connectivity list of all elements of heart model                                                    %%% 
%%%              .FibreOrientation  ->  x,y,z components of a unit vector defining the fibre orientation for each element of heart model    %%% 
%%%              .Material          ->  Label defining which material each element of heart model belongs to                                %%% 
%%%              .LongCV            ->  Longitudinal conductivity for each material existing in heart model                                 %%% 
%%%              .AnisotropyRatio   ->  Anisotropy ratio (trans. CV/long. CV) for each material existing in heart model                     %%% 
%%% --------------------------------------------------------------------------------------------------------------------------------------- %%%

% Load torso files
fprintf ('\n    ->  Reading TORSO files ... \n');
if nargin < 5
    organID = [];
end
[torsoElements, torsoNodes] = readTorsoModelFile (torsoModelFile, tissueFile, myocardLabel, organID);
%%% ---------------------------------------------------------------------------------------------------------------------------------------------- %%% 
%%%   torsoElements.NumElements       ->  Number of elements (tetrahedra) in torso model                                                           %%% 
%%%                .ConnectivityList  ->  Connectivity list of elements (tetrahedra) of torso model                                                %%%
%%%                .LongCV            ->  Longitudinal conductivity for each element of torso model                                                %%%
%%%                .AnisotropyRatio   ->  Anisotropy ratio (transveral CV/longitudinal CV) for each element of torso model                         %%%
%%%                .OrganID           ->  Label defining to which organ/tissue belongs each element of torso model                                 %%%
%%%                .FibreOrientation  ->  x,y,z coordinates of the unit vector defining the fibre orientation for each element of torso model      %%%
%%%                .MyoID             ->  Value of 'OrganID' label corresponding to 'myocardium' (ventricles and/or atria)                         %%%
%%%                .NumMyoElements    ->  Number of elements of torso model labeled as 'myocardium'                                                %%%
%%%                .MyoElements       ->  Boolean array indicating those elements of torso model labeled as 'myocardium'                           %%%
%%% ---------------------------------------------------------------------------------------------------------------------------------------------- %%%
%%%   torsoNodes.NumNodes     ->  Number of nodes in torso model                                                                                   %%%
%%%             .Coordinates  ->  x,y,z coordinates of all nodes of torso model                                                                    %%% 
%%%             .NumMyoNodes  ->  Number of nodes of torso model belonging to elements labeled as 'myocardium'                                     %%% 
%%%             .MyoNodes     ->  Boolean array indiciating those nodes of torso model that belong to elements labeled as 'myocardium'             %%% 
%%% ---------------------------------------------------------------------------------------------------------------------------------------------- %%%

% Checking mesh
fprintf ('    ->  Checking TORSO MODEL mesh. Looking for orphan nodes ... \n');
conList = torsoElements.ConnectivityList;
conList = unique (conList);
nodesIDs = false (torsoNodes.NumNodes, 1);
nodesIDs (conList) = true;
% Check whether all nodes belong at least to one element of torso model mesh
if all(nodesIDs)   % if so
    fprintf ('           *  PERFECT. No orphan node was found \n');
else   % if there are orphan nodes
    indOrphan = find (nodesIDs);
    fprintf ('\n\n\t\t\t WARNING  :::  generateFilesForTorsoPropagation ');
    fprintf ('\n\t\t\t                   -  There are ORPHAN nodes in the mesh of TORSO MODEL');
    fprintf ('\n\t\t\t                   -  Please, check mesh !!!!!!');
    fprintf ('\n\t\t\t                   -  List of orphan nodes: \n\n');
    fprintf ('\t\t\t                            %7d \n', indOrphan);
    fprintf ('\n\n');
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  ********************************************************************* %
%%% --  PARALLEL COMPUTING  --------------------------------------------- %
%%% ********************************************************************* %
tic
disp (' ')
disp (' PARALLEL COMPUTING:')
% Check whether parallel pool is already open
if ~isempty (gcp ('nocreate'))  % if so
    % Close it
    evalc ('delete (gcp)');
end  % if NOT, just continue
% Check OS of the system and get the number of processors available in machine
if isunix  % if UNIX-based system
    [~, numCPUs] = system ('nproc');         % for LINUX/UNIX systems
elseif ispc   % if Windows running on PC
    numCPUs = getenv ('NUMBER_OF_PROCESSORS');  % for WINDOWS systems
end
fprintf ('   - Number of cores available for parallel computing:  %s \n', numCPUs);
numCPUs = str2double (numCPUs);
% Set 'numCPUs' as the maximum allowed number of workers in 'local' profile for parallel computing
parOptions = parcluster ('local');
parOptions.NumWorkers = numCPUs;
% Open the parallel pool
msg = evalc ('pool = parpool (''local'', numCPUs);');
pool.IdleTimeout = Inf;   % Set 'No automatic shut down'
disp (['   - ', msg(1:end-2)])
% Execution time
disp (['   - Elapsed time for STARTING PARALLEL POOL:  ' executionTime(toc)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  ********************************************************************* %
%%% --  COMPUTATION OF RADIO FOR NEIGHBOURS SEARCHES  ------------------- %
%%% ********************************************************************* %
tic
fprintf ('\n    ->  Calculating mean length of edges of HEART MODEL to use it as radio for neighbour search ...\n');
fprintf ('           *  Getting connectivity list of all edges of HEART MODEL:    ');
% Create a new VTKstruct
heartModel = createVTKstruct ();
heartModel.Dataset = 'UNSTRUCTURED_GRID';
heartModel.Points = elviraModel.Coordinates;
heartModel.NumPoints = elviraModel.NumNodes;
% Assuming all elements in 'heartModel' are HEXAHEDRA
heartModel.NumCells = elviraModel.NumElements;
numNodesPerCell = size (elviraModel.ConnectivityList, 2);
heartModel.Cells = cell (heartModel.NumCells, 2);
heartModel.Cells(:,1) = num2cell (numNodesPerCell*ones(heartModel.NumCells,1), 2);
heartModel.Cells(:,2) = (num2cell ((elviraModel.ConnectivityList-1)', 1))';  %%% Subtract '1' to 'elm_src' to transform Matlab/Elvira indices starting at '1' into VTK indices starting at '0'
heartModel.CellTypes = 12 * ones (heartModel.NumCells, 1);  %%%  'VTK_HEXA = 12' -> '12' is the VTK code for hexahedral cell/element
% Get edges
edgesConList = findEdges (heartModel);
fprintf ('%s \n', executionTime(toc));
% Compute edges length
tic
fprintf ('           *  Computing edges length:    ');
edgesLength = getEdgesLength (edgesConList, heartModel);
fprintf ('%s \n', executionTime(toc));
% Establish initial search radio as the average of edges legnth of HEART MODEL 
radInit = mean (edgesLength);
fprintf ('           *  Average of edges length of HEART MODEL:    %.4f \n', radInit);
radInit = sqrt(3) * radInit;
fprintf ('           *  Initial search radio {''radInit''}:    %.4f \n', radInit);
% Remove some variables to free memory
clear heartModel numNodesPerCell edgesConList edgesLength;


%%  ********************************************************************* %
%%% --  MAPPING FIBRE ORIENTATION FROM HEART MODEL TO TORSO MODEL  ------ %
%%% ********************************************************************* %
% Material properties for heart elements in torso -> Add fibre orientation
fprintf ('\n    ->  Assigning material properties to myocardial elements in torso ...\n');
tic
fprintf ('           *  Getting conectivity list of cells/elements of TORSO MODEL labeled as ''myocardium'':    ');
% Get the connectivity list of all cells/elements (tetrahedra) of torso model labeled as 'myocardium'
conListMyoCells = torsoElements.ConnectivityList (torsoElements.MyoElements, :);
% Reshape the list into a column vector
conListMyoCells = reshape (conListMyoCells', numel(conListMyoCells), 1);
% Take the x,y,z coordinates of points/nodes in the list
myoNodesCoord = torsoNodes.Coordinates (conListMyoCells, :);
% Break up the list into a cell array where each array element stores the x,y,z coordinates of the 4 nodes that form a given tetrahedron
conListMyoCells = mat2cell (myoNodesCoord, 4*ones(torsoElements.NumMyoElements,1), 3);
% Remove some variables in order to free memory
clear myoNodesCoord;
fprintf ('%s \n', executionTime(toc));
tic
fprintf ('           *  Computing centroids of elements of HEART MODEL:    ');
% Compute centroids of cells/elements of heart model (ventricles or atria)
heartCentroids = zeros (elviraModel.NumElements, 3);
% Go through all cells/elements in HEART MODEL
parfor i = 1:elviraModel.NumElements
    % Calculate centroid of each cell/element
    heartCentroids (i,:) = mean (elviraModel.Coordinates (elviraModel.ConnectivityList(i,:), :), 1); %#ok<PFBNS>
end
fprintf ('%s \n', executionTime(toc));
% Assign the conductivity along the fibre and the conductivity ratio to the ventricles
tic
fprintf ('           *  Mapping FIBRE ORIENTATION from HEART MODEL to TORSO MODEL:    ');
% Initialise some variables
fibresMyo = zeros (torsoElements.NumMyoElements, 3);
longConduct = zeros (torsoElements.NumMyoElements, 1);
ratioTransLong = zeros (torsoElements.NumMyoElements, 1);
distClosestHeartCell = zeros (torsoElements.NumMyoElements, 1);
% Go through all cells/elements of torso model labeled as 'myocardium'
parfor i = 1:torsoElements.NumMyoElements
    % Compute centroid of current cell/element of torso model
    Xc = mean (conListMyoCells{i});
    % Find the cell/element of HEART MODEL which is the the closest to the current cell/element (tetrahedron) of TORSO MODEL
    [indClosest, radSearch, dist] = find_neighbours (Xc, heartCentroids, radInit);
    % Once the closest cell/elements of HEART MODEL has been found
    % Calculate longitudinal conductivity ('sigma') taking into account the value from Elvira's material file
    longConduct (i) = (1 + lambda) * elviraModel.LongCV(elviraModel.Material(indClosest)); %#ok<PFBNS>
    % And also calculate the ratio between transversal and longitudinal conductivities ('a')
    ratioTransLong (i) = elviraModel.AnisotropyRatio(elviraModel.Material(indClosest));
    % Define fibre orientation
    fibresMyo (i, :) = elviraModel.FibreOrientation (indClosest, :);
    % Check whether a cell/element of HEART MODEL was found into a vicinity defined by 'radInit'
    if radSearch > radInit  % if NOT
        % Save the distance from current torso node to the closest node of
        % cell/element of heart model taken for interpolation
        distClosestHeartCell (i) = dist;
    end  % if so, continue
end

% Save conductivities and fibre orientation in 'torsoElements' structure
% Fibre orientation
torsoElements.FibreOrientation = zeros (torsoElements.NumElements, 3);
torsoElements.FibreOrientation(torsoElements.MyoElements,:) = fibresMyo (:,:);
% Longitudinal conductivity
torsoElements.LongCV(torsoElements.MyoElements) = longConduct;
% Anisotropy ratio
torsoElements.AnisotropyRatio = ones (torsoElements.NumElements, 1);          %%%  Anisotropy ratio (transveral CV / longitudinal CV)
torsoElements.AnisotropyRatio(torsoElements.MyoElements) = ratioTransLong;    %%%     ->  All organs/tissue show isotropic propagation except for the 'myocardium'
% Remove variables to free memory
clear conListMyoCells fibresMyo;
fprintf ('%s \n', executionTime(toc));

% Report about interpolation process
fprintf ('\n    ->  Report about fibre orientation mapping process ...\n');
% Look for torso cells/elements for which NO cell/element of heart model was found into the vicinity defined by 'radInit'
indMyoElements = find (torsoElements.MyoElements);
nodeInd = indMyoElements (distClosestHeartCell > 0);
distClosestHeartCell = distClosestHeartCell (distClosestHeartCell > 0);
if ~isempty (nodeInd)
    fprintf ('\n           *  WARNING. There are %d TORSO CELLS/ELEMENTS (of a total of %d) for which NO cell/element of HEART MODEL was found into a vicinity defined by ''radInit'' \n\n', ...
                length(nodeInd), torsoElements.NumMyoElements);
    fprintf ('\t\t\t\t Torso cell/elem. index \t\t Dist to centroid of the closest heart model element \n');
    for j = 1:length(nodeInd)
        fprintf ('\t\t\t        %6d     \t\t\t         %.3f \n', nodeInd(j), distClosestHeartCell(j));
    end
else
    fprintf ('           *  PERFECT !!! For all TORSO CELLS/ELEMENTS labeled as ''myocardium'' a cell/element of HEART MODEL was found into the vicinity defined by ''radInit'' \n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  ********************************************************************* %
%%% --  INTERPOLATION FOR MYOCARDIUM NODES  ----------------------------- %
%%% ********************************************************************* %
% Defining interpolation for myocardial target nodes
tic
fprintf ('\n    ->  Defining interpolation on MYOCARDIUM nodes of TORSO MODEL ... \n');
% Take the x,y,z coordinates of torso nodes labeled as 'myocardium'
myoNodesCoord = torsoNodes.Coordinates(torsoNodes.MyoNodes,:);
distClosestHeartCell = zeros (torsoNodes.NumMyoNodes, 1);
interpol = struct ('Id', [], 'alpha', []);
% Go through all points/nodes of torso model belonging to cells/elements labeled as 'myocardium'
parfor i = 1:torsoNodes.NumMyoNodes
    % Take the x,y,z coordinates of the current node of torso model
    torsoNodeCoord = myoNodesCoord(i,:);
    % Find the centroid of heart model which is the closest to the current node of torso model
    % The idea is to find the cell/element of heart model that contains the current node of torso model
    [indClosest, radSearch, distCentroid] = find_neighbours (torsoNodeCoord, heartCentroids, radInit);
    % Get the connectivity list of the cell/element of heart model corresponding to the closest centroid
    conListClosest = elviraModel.ConnectivityList(indClosest,:); %#ok<PFBNS>
    coordNodesHeartCell = elviraModel.Coordinates(conListClosest,:);
    % Compute the euclidean distance from current node of torso model to each node that from the cell/element of heart model
    dist = pdist2 (torsoNodeCoord, coordNodesHeartCell, 'euclidean');
    % Get the interpolation coefficients based on the distance from current
    % node of torso model to each node of the cell/element of heart model 
    alpha = inter_coef (dist);
    interpol(i).Id = conListClosest'; %#ok<PFOUS>
    interpol(i).alpha = alpha';
    % Check whether a cell/element of HEART MODEL was found into a vicinity defined by 'radInit'
    if radSearch > radInit  % if NOT
        % Save the distance from current torso node to the closest node of
        % cell/element of heart model taken for interpolation
        distClosestHeartCell (i) = distCentroid;
    end  % if so, continue
end
fprintf ('           *  Elapsed time:   %s \n', executionTime(toc));

% Report about interpolation process
fprintf ('\n    ->  Report about the interpolation process ...\n');
% Look for torso nodes for which NO cell/element of heart model was found into the vicinity defined by 'radInit'
nodeInd = indMyoElements (distClosestHeartCell > 0);
distClosestHeartCell = distClosestHeartCell (distClosestHeartCell > 0);
if ~isempty (nodeInd)
    fprintf ('\n           *  WARNING. There are %d TORSO NODES (of a total of %d) for which NO cell/element of HEART MODEL was found into a vicinity defined by ''radInit'' \n\n', length(nodeInd), torsoNodes.NumMyoNodes);
    fprintf ('\t\t\t\t Torso node index \t\t Dist to centroid of the closest heart model element \n');
    for j = 1:length(nodeInd)
        fprintf ('\t\t\t        %6d     \t\t\t         %.3f \n', nodeInd(j), distClosestHeartCell(j));
    end
else
    fprintf ('           *  PERFECT !!! For all TORSO NODES labeled as ''myocardium'' a cell/element of HEART MODEL was found into the vicinity defined by ''radInit'' \n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Shut down parallel pool
evalc ('delete (pool);');

% Save data for torso propagation in a .mat file
tic
fprintf('\n    ->  Saving data for torso propagation in a .mat file ... \n');
save (dataFileName, 'torsoElements', 'torsoNodes', 'interpol');  %%% , '-v7.3');
fprintf('\t\t\t *  Elapsed time:  %s \n', executionTime(toc));

% Final message
fprintf('\n    ->  FINISH !!!!!!!!!!!!!!\n');
fprintf('\t\t\t *  Elapsed time for whole process:  %s \n\n', executionTime(toc(tStart)));

% Check whether it runs on a Matlab session or in deployed mode
if ~isdeployed  % if Matlab session
    % Close the log file
    diary;
    diary off;
end   % if deployed mode, just continue

% return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  AUXILIARY FUNCTIONS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find neighbours to X in Xneig within radius rad, and the distance from each neighbourg to X
% function [indClosest, radSearch, dist] = find_neighbours (X, Xneig, radInit)
% 
% % Initialise variables
% indIntoBB = [];
% radSearch = 0;
% 
% while isempty (indIntoBB)
%     % Update search radio
%     radSearch = radSearch + radInit;
%     
%     %%% X axis
%     % Define search range for X coordinate
%     xRange = [X(1)-radSearch, X(1)+radSearch];
%     % Take points whose value for X coordinate is within the range specified by 'xRange'
%     indX = (Xneig(:,1) >= xRange(1)) & (Xneig(:,1) <= xRange(2));
%     % Take the indices of points in 'Xreig' satisfying 'xRange'
%     indIntoBB = find (indX);
%     pointsIntoBB = Xneig (indIntoBB, :);
%     
%     % Check whether some points satisfy previous conditions
%     if ~isempty (indIntoBB)
%         %%% Y axis
%         % Define search range for Y coordinate
%         yRange = [X(2)-radSearch, X(2)+radSearch];
%         % Take points whose value for Y coordinate is within the range specified by 'yRange'
%         indY = (pointsIntoBB(:,2) >= yRange(1)) & (pointsIntoBB(:,2) <= yRange(2));
%         % Take the indices of points in 'Xreig' satisfying 'xRange' and 'yRange'
%         indIntoBB = indIntoBB (indY);
%         pointsIntoBB = Xneig (indIntoBB, :);
%         
%         % Check whether some points satisfy previous conditions
%         if ~isempty (indIntoBB)
%             %%% Z axis
%             % Define search range for Z coordinate
%             zRange = [X(3)-radSearch, X(3)+radSearch];
%             % Take points whose value for Z coordinate is within the range specified by 'zRange'
%             indZ = (pointsIntoBB(:,3) >= zRange(1)) & (pointsIntoBB(:,3) <= zRange(2));
%             % Take the indices of points in 'Xreig' satisfying 'xRange', 'yRange' and 'zRange'
%             indIntoBB = indIntoBB (indZ);
%             pointsIntoBB = Xneig (indIntoBB, :);
%             
%             %%% Distance
%             % Check whether some points satisfy previous conditions
%             if ~isempty (indIntoBB)
%                 % Calculate distance to the closest node
%                 [dist, ind] = pdist2 (pointsIntoBB, X, 'euclidean', 'Smallest', 1);
%                 % Check whether the distance to the closest node into bounding box is smaller or equal to 'radSearch'
%                 if dist <= radSearch  % if so
%                     % Take the index of the closest node
%                     indClosest = indIntoBB (ind);
%                 else  % if it is greater
%                     % Set 'indIntoBB' as an empty matrix to keep searching
%                     indIntoBB = [];
%                 end
%             end
%         end
%     end
% end
% 
% return


% %%% --------------------------------------
% % Calculates interpolating coefficients based on radial function principle
% function alpha = inter_coef (dist)
% ncoef = length(dist);
% dist = dist/mean(dist);
% alpha = ones(ncoef,1);
% for i = 1:ncoef
%     for j = 1:ncoef
%         if (j ~= i)
%             alpha(i) = alpha(i) * dist(j);
%         end
%     end
% end
% denominator = sum(alpha);
% alpha = alpha/denominator;
% return