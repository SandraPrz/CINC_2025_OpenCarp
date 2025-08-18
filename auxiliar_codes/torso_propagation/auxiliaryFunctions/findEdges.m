%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    [edges, repetitions, VTKstruct] = findEdges (VTKstruct, append2VTKstruct)     %%%          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  'findEdges' returns in 'edges' the connectivity list of all the edges  %
%  existing in a mesh, which can be a triangle-based surface mesh or a    %
%  volume mesh based on voxels, hexahedra or tetrahedra                   %
%     IMPORTANT. The indices returned in 'edges' starts at '1' as Matlab  %
%                requires, NOT at '0' as VTK does                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ---------------------------------------------------------------------------------------------------------------------- %
%     VTKstruct         ->  Matlab structure containing all the information extracted from a .VTK file by means of the
%                           function 'vtk2structReader'
%     append2VTKstruct  ->  [ OPTIONAL ]
%                           * If this input argument is set to any of the following values {'y','yes',1,true}, the edges
%                             length wil be calculated and three new CELL_DATA fields will be added to 'VTKstruct' in order
%                             to save the length of the largest and the shortest edge belonging to each cell/element of
%                             the mesh and the ratio between them (largest/shortest), respectively.
%                           * Otherwise, even when it is omitted, edges length will not be calculated and hence no new CELL_DATA
%                             field will be added to 'VTKstruct'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUTs -------------------------------------------------------------------------------------------------------------------- %
%     edges      ->  There are two possibilities for this output:
%                       * OPTION 1. If the option 'append2VTKstruct' is disabled, 'edges' will be a Nx2 matrix containing
%                                   the connectivity list (1) for all the edges existing in the mesh, where N is the total
%                                   number of edges
%                       * OPTION 2. If the option 'append2VTKstruct' is enabled, 'edges' will be a Matlab structure including
%                                   two fields: 'Connectivity', which will contain the same Mx2 matrix explained in OPTION 1,
%                                   and 'Length', which will contain the length of all the edges defined by the returned
%                                   connectivity list
%   repetitions  ->  Array defining how many times each edge is repeated, i.e., how many cells/elements each edge belongs to
%     VTKstruct  ->  If the option 'append2VTKstruct' is enabled 'VTKstruct' will be returned including the two
%                    aforementioned CELL_DATA fields
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMPORTANT REMARKS  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
%     (1) The indices returned in 'edges' starts at '1' as Matlab requires, NOT at '0' as VTK does.
%     (2) 'findEdges' uses parallel computing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP  --------------------------------------------------------- %
%                Author:    Alejandro Daniel Lopez Perez
%         Creation date:    ??/??/2015
%     Last modification:    19/10/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [edges, repetitions, VTKstruct] = findEdges (VTKstruct, append2VTKstruct)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CHECK INPUT ARGUMENT FOR 'append2VTKstruct'  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 2
    append2VTKstruct = false;
else
    switch append2VTKstruct
        case {'y', 'yes', 1, true}
            append2VTKstruct = true;
        otherwise
            append2VTKstruct = false;
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%
%%%  CELL/ELEMENT TYPE  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check which kind of mesh it is
switch VTKstruct.Dataset
    
    case 'POLYDATA'
        % Check whether the cells/elements are triangles
        switch size (VTKstruct.Cells, 2)
            case 0  % if there is no elements (points cloud instead of a mesh)
                warning ('findEdges  :::  No element defined in this POLYDATA VTK file');
                edges = [];
                return
            case 3  % if TRIANGLES
                elemType = 'triangle';
            otherwise  % if any other kind of element different from triangles
                warning ('findEdges  :::  Unsupported type of element (non-triangles) for VTK POLYDATA meshes');
                edges = [];
                return
        end
    
    case 'UNSTRUCTURED_GRID'
        % Check whether the cells/elements are voxels (VTK_VOXEL = 11) or
        % hexahedra (VTK_HEXAEDRON = 12)
        switch VTKstruct.CellTypes(1)
            case 10  % if TETRAHEDRA
                elemType = 'tetra';
            case 11  % if VOXELS
                elemType = 'voxel';
            case 12  % if HEXAHEDRA
                elemType = 'hexahedron';
            otherwise
                warning ('findEdges  :::  Unsupported type of element for VTK UNSTRUCTURED_GRID meshes');
                edges = [];
                return
        end
    
    otherwise
        warning ('findEdges  :::  Unsupported VTK DATASET');
        edges = [];
        return
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SEARCH ALL EDGES FOR EACH CELL/ELEMENT  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check the cell/element type to be assessed 
switch elemType
    case 'triangle'
        % Set the order in which the nodes of each element must be combined
        % to get all edges that form it
        listFirstNode  = [1, 1, 2];   % '1st' node for each edge of the element
        listSecondNode = [2, 3, 3];   % '2nd' node for each edge of the element
        % Get the whole connectivity list for all cells/elements {triangles} of the evaluated mesh
        conList = double (VTKstruct.Cells) + 1;  %%% Add 1 because VTK indices start at '0' whereas in Matlab start at '1'
        % Set the number of edges per cell/element
        numEdgesPerCell = 3;
    case 'tetra'
        % Set the order in which the nodes of each element must be combined
        % to get all edges that form it
        listFirstNode  = [1, 1, 1, 2, 2, 3];   % '1st' node for each edge of the element
        listSecondNode = [2, 3, 4, 3, 4, 4];   % '2nd' node for each edge of the element
        % Get the whole connectivity list for all cells/elements {tetrahedra} of the evaluated mesh
        conList = double ([VTKstruct.Cells{:,2}]') + 1;  %%% Add 1 because VTK indices start at '0' whereas in Matlab start at '1'
        % Set the number of edges per cell/element
        numEdgesPerCell = 6;
    case 'voxel'
        % Set the order in which the nodes of each element must be combined
        % to get all edges that form it
        listFirstNode  = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7];   % '1st' node for each edge of the element
        listSecondNode = [2, 3, 5, 4, 6, 4, 7, 8, 6, 7, 8, 8];   % '2nd' node for each edge of the element
        % Get the whole connectivity list for all cells/elements {voxels} of the evaluated mesh
        conList = double ([VTKstruct.Cells{:,2}]') + 1;  %%% Add 1 because VTK indices start at '0' whereas in Matlab start at '1'
        % Set the number of edges per cell/element
        numEdgesPerCell = 12;
    case 'hexahedron'
        % Set the order in which the nodes of each element must be combined
        % to get all edges that form it
        listFirstNode  = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7];   % '1st' node for each edge of the element
        listSecondNode = [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8];   % '2nd' node for each edge of the element
        % Get the whole connectivity list for all cells/elements {hexahedra} of the evaluated mesh
        conList = double ([VTKstruct.Cells{:,2}]') + 1;  %%% Add 1 because VTK indices start at '0' whereas in Matlab start at '1'
        % Set the number of edges per cell/element
        numEdgesPerCell = 12;
end

% Construct the connectivity list for all the edges that form each element of the mesh
%   (allEdges -> Nx2 matrix, where N=M*P, where M is the number of edges per element and P is the total number of elements in the mesh)
node0 = conList (:,listFirstNode)';                 % First, create the list of the '1st' node of all of edges in the mesh
allEdges = reshape (node0, numel(node0), 1);        % And reshape it into a column vector
node1 = conList (:,listSecondNode)';                % Now, create the list of the '2nd' node of all of edges in the mesh
allEdges (:,2) = reshape (node1, numel(node1), 1);  % And reshape it into a column vector setting it as the 2nd column in the connectivity list
clear node0 node1;  % Remove some useless variables to free memory

% Check whether the option 'append2VTKstruct' is enabled
if append2VTKstruct  % if so
    
    %%% LARGEST_EDGE
    % Add a new CELL_DATA fields to 'VTKstruct' for the "largestEdge" of each cell/element
    VTKstruct = setCellDataField (VTKstruct);
    % Set the field header
    indLargestEdge = length (VTKstruct.CellData);
    VTKstruct.CellData(indLargestEdge).Type = 'SCALARS';
    VTKstruct.CellData(indLargestEdge).Name = 'largestEdge';
    VTKstruct.CellData(indLargestEdge).Format = 'float';
    % Create an auxiliary variable to save the data associated to this field
    largestEdge = zeros (VTKstruct.NumCells, 1);
    
    %%% SHORTEST_EDGE
    % Add another CELL_DATA fields to 'VTKstruct' for the "shortestEdge" of each cell/element
    VTKstruct = setCellDataField (VTKstruct);
    % Set the field header
    indShortestEdge = indLargestEdge + 1;
    VTKstruct.CellData(indShortestEdge).Type = 'SCALARS';
    VTKstruct.CellData(indShortestEdge).Name = 'shortestEdge';
    VTKstruct.CellData(indShortestEdge).Format = 'float';
    % Create an auxiliary variable to save the data associated to this field
    shortestEdge = zeros (VTKstruct.NumCells, 1);
    
    %%% EDGE_RATIO
    % Add another CELL_DATA fields to 'VTKstruct' for the "shortestEdge" of each cell/element
    VTKstruct = setCellDataField (VTKstruct);
    % Set the field header
    indEdgeRatio = indLargestEdge + 2;
    VTKstruct.CellData(indEdgeRatio).Type = 'SCALARS';
    VTKstruct.CellData(indEdgeRatio).Name = 'edgeRatio';
    VTKstruct.CellData(indEdgeRatio).Format = 'float';
    
    %%% EDGES LENGTH
    % Calculate the length for all the edges belonging to each cell/element
    allEdgesLength = getEdgesLength (allEdges, VTKstruct.Points);
    % Open the parallel pool, only if it is not opened yet
    evalc ('gcp()');
    % Look for the largest and the shortest edge for each cell/element
    parfor j = 1:VTKstruct.NumCells
        % Retrieve the length of all the edges belonging to the current cell/element
        lengthEdgesCurrentCell = allEdgesLength (1+((j-1)*numEdgesPerCell) : j*numEdgesPerCell);
        % Get the largest one
        largestEdge (j) = max (lengthEdgesCurrentCell);
        % Get the shortest one
        shortestEdge (j) = min (lengthEdgesCurrentCell);
    end
end

% Sort the connectivity list in order to set the node with the lower index
% in the first column of the connectivity list
allEdges = sort (allEdges, 2, 'ascend');
% Now sort the first column of the whole connectivity list in 'ascending' order
[allEdges(:,1), I] = sort(allEdges(:,1), 'ascend');
% And apply the same reordering to the second column to not alter the connectivity list
allEdges (:,2) = allEdges(I, 2);
% And also to the 'allEdgesLength' array, ...
if append2VTKstruct  % ... only if the option 'append2VTKstruct' is enabled
    allEdgesLength = allEdgesLength (I);
else
    % It seems to make no sense, but if 'allEdgesLength' is not defined the
    % parfor loop goes crazy, even though it does not use this variable
    % when the option 'append2VTKstruct' is disabled
    allEdgesLength = [];
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  REMOVE REPEATED EDGES  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% As the most of edges belongs to more than one cell/element, the following
%%% step is to remove from the list all those edges that appear more than
%%% once, i.e., remove all repeated appearances of the same edge
% First, look for groups of edges that share the same node as the '1st' one
indNodeGroups = find (diff(allEdges(:,1)));
indNodeGroups = [0; indNodeGroups; size(allEdges,1)];

% Create a cell array to save the connectivity list for all those edges
% that share a given node with no repetitions
edgesPerNode = cell (length(indNodeGroups)-1, 1);
% Create a cell array to save how many times each edge is repeated, i.e.,
% the number of cells/elements to which each edge belongs
repetitionsPerEdge = cell (length(indNodeGroups)-1, 1);
% Check whether the option 'append2VTKstruct' is enabled
if append2VTKstruct  % if so
    edgesLength = cell (length(indNodeGroups)-1, 1);
end

% Remove repeated edges from the list
parfor i = 1:length(indNodeGroups)-1
    % Get the shared node for the current group of edges
    sharedNode = allEdges(indNodeGroups(i+1),1);
    % Get the list of the '2nd' nodes for all edges in this group, also
    % removing the repeated nodes and sorting them in 'ascending' order
    [secondNodes, indUnique, ~] = unique (allEdges(indNodeGroups(i)+1:indNodeGroups(i+1), 2)); %#ok<*PFBNS>
    % And save the new connectivity list for this group, which is sorted and with no repetitions
    edgesPerNode{i} = [sharedNode*ones(length(secondNodes),1), secondNodes];
    % Count how many times each edge in current group is repeated whitin the group
    repetitionsPerEdge{i} = zeros (length(secondNodes), 1);
    for k = 1:length(secondNodes)
        % Count how many times each second node apperas in current group
        repetitionsPerEdge{i}(k) = sum (allEdges(indNodeGroups(i)+1:indNodeGroups(i+1), 2) == secondNodes(k));
    end
    % Check whether the option 'append2VTKstruct' is enabled
    if append2VTKstruct  % if so
        % Save the length for the non-removed edges in the current group
        aux = allEdgesLength(indNodeGroups(i)+1:indNodeGroups(i+1));
        edgesLength{i} = aux (indUnique);
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SAVE FINAL RESULTS  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check once again whether the option 'append2VTKstruct' is enabled
if append2VTKstruct  % if so
    % Define 'edges' as a Matlab structure
    edges = struct ('Connectivity', [], 'Length', []);
    % Transform the resulting 'edgesPerNode' list (cell array) into a single matrix
    % which correponds to the connectivity list of all edges existing in the mesh
    %%% Note that this action also remove all empty cells from the cell array
    edges.Connectivity = cell2mat (edgesPerNode);
    % Save the edges length
    edges.Length = cell2mat (edgesLength);
    % Save the data for the largest and the shortest edges and the ratio between them
    % in its corresponding CELL_DATA fields into 'VTKstruct'
    VTKstruct.CellData(indLargestEdge).Data = largestEdge;
    VTKstruct.CellData(indShortestEdge).Data = shortestEdge;
    VTKstruct.CellData(indEdgeRatio).Data = largestEdge ./ shortestEdge;
else
    % Only save the connectivity list in 'edges'
    edges = cell2mat (edgesPerNode);
    clear VTKstruct
end
% Convert 'repetitions' cell array into a simple array
repetitions = cell2mat (repetitionsPerEdge);