%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%      fullPath = setNewFileName (path, fileName)       %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    ...
%    ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fullPath = setNewFileName (path, fileName)

% Check the inputs
switch nargin
    case 1  % if only one input
        % Assume the passed argument is the 'fullPath'
        fullPath = path;
    case 2  % if two inputs
        % Generate the 'fullPath' from 'path' and 'fileName'
        fullPath = fullfile (path, fileName);
end

% Check whether the proposed file name already exists
if ~exist (fullPath, 'file')  % if it does not exist
    return
end  % if it already exists, continue

% Get the file extension
extension = getFileExtension (fullPath);
% Check the extension
if ~isempty (extension)
    fullPath = fullPath (1:end-length(extension));
end

% Generate a new file name in order not to overwrite and therefore remove
% the already existing one
ind = 1;
newFileName = strcat (fullPath, '(1)', extension);
while exist (newFileName, 'file')
    ind = ind + 1;
    newFileName = strcat (fullPath, '(', num2str(ind), ')', extension);
end
fullPath = newFileName;