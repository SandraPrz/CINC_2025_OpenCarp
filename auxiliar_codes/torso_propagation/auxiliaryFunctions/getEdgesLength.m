%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%                                                 %%%%%%%%%%%%%
%%%%%%    edgesLength = getEdgesLength (edgesConList, VTKstruct)     %%%%%%
%%%%%%%%%%%%%                                                 %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  'edgeLength' returns the length of the edges corresponding to the mesh %
%  contained in 'VTKstruct' and defined by 'edgesConList'                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs  -------------------------------------------------------------------------------------------- %
%     edgesConList  ->  Nx2 matrix corresponding to the connectivity list
%                       of the edges whose length must be measure 
%     points        ->  There are two options for this argument:
%                         * OPTION 1. Matlab structure containing all the information extracted
%                                     from a .VTK file by means of the function 'vtk2structReader'
%                         * OPTION 2. Nx3 matrix containig the X,Y,Z coordinates of the nodes/points,
%                                     where N is the number of points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUTs  ------------------------------------------------------------------------------------------- %
%     edgesLength  ->  Length of each edge defined by 'edgesConList' in terms of Euclidean distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edgesLength = getEdgesLength (edgesConList, points)

% Check whether input 'points' is a Matlab structure
if isstruct (points)  % if so
    % Extract the field 'Points' from the 'VTKstruct'
    aux = points.Points;
    clear points
    points = aux;
    clear aux
end

%%% Get the coordinates of the points/nodes that form each defined edge by 'edgesConList'
% -> 1st node of each edge
node0 = points (edgesConList(:,1), :);
% -> 2nd node of each edge
node1 = points (edgesConList(:,2), :);

% Calculate the edges length by applying the Euclidean distance formula
edgesLength = sqrt (sum(((node0-node1).^2),2));