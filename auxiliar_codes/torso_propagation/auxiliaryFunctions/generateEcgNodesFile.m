%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ------------------------------------------------------------ %%%
%      surfMeshFile  ->  String containing the full path of a VTK file
%                        containing a surface mesh generated using
%                        ParaView as described below.
%%%   'surfMeshFile' mesh generation process:
%%%         In ParaView apply following filters on volume mesh
%%%         of torso model:
%%%             ->  Generate IDs
%%%             ->  Extract Surface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function generateEcgNodesFile (surfMeshFile)

% Read VTK file containing the torso surface
torsoSurf = vtk2structReader (surfMeshFile);
% Check whether there is any FIELD in the VTK file
if ~isempty (torsoSurf.Field)  % if so
    torsoSurf = field2data (torsoSurf);
end
% Get original points/nodes indices
pointIDs=torsoSurf.PointData.Data;
%[~, pointIDs] = findPointDataByName (torsoSurf, 'Ids');
ecgNodes = pointIDs + 1;   %#ok<NASGU>   %%% Add '1' because Matlab indices start at '1' while VTK indices start at '0'
% % % ecgNodes = sort (pointIDs + 1);   %#ok<NASGU>   %%% Add '1' because Matlab indices start at '1' while VTK indices start at '0'
% Save 'ecgNodes' in a .mat file
nodesFileName = setNewFileName (strcat (surfMeshFile(1:end-4), '_ECG_NODES.mat'));   %%% In order not to overwrite an already existing file 
save (nodesFileName, 'ecgNodes');