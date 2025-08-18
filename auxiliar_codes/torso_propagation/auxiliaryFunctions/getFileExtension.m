%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%       extension = getFileExtension (fileName)         %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Given a file name, 'getFileExtension' returns the extension of the
%    file. If the file has no extension, an empty matrix will be returned
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function extension = getFileExtension (fileName)
% Look for dots into the string containing the file name
dots = strfind (fileName, '.');
% Check whether there is at least one dot
if isempty (dots)  % if NOT
    extension = [];
else  % if so
    % Assume the extension is the final part of the string from the
    % position of the last found dot
    extension = fileName (dots(end):end);
end