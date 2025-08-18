function V = readPotentialFile (potFilePath)
% Get the extension of potential file
[~, ~, extension] = fileparts (potFilePath);
% And check the type of file that contains potential data
switch extension
  % if .MAT files
    case '.mat'
        % Load .mat file
        V = load (potFilePath);
        varName = fieldnames (V);
        V = V.(varName{1});
  % if .ENS files
    case '.ens'
        % Open .ens file
        fidPot = fopen (potFilePath, 'r');
        % Look for the 'coordinates' section into the .ens file
        while ~feof(fidPot)
            if ~isempty (strfind(fgetl(fidPot), 'coordinates'))
                break;
            end
        end
        % Get potential values
        V = cell2mat (textscan (fidPot, '%f', Inf));   %%% faster than using 'fscanf'
%%%%        V = fscanf (fidPot, '%f', Inf);
        % Close .ens file
        fclose (fidPot);
end
