%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   step3_runTorsoPropagation (matricesFile, potentialFilesFolder, ecgNodeFile, output_id, parallel, solver, tol_pcg, maxIter)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%      3rd (and last) STEP IN TORSO PROPAGATION PROCESS      %%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    'step3_runTorsoPropagation' computes the torso propagation using the specified
%    solver. As an output a .mat file, containing the extracellular potentials
%    computed for those nodes of torso model specified by the input 'ecgNodeFile',
%    will be created in the same folder of the input 'matricesFile'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ------------------------------------------------------------ %%%
%     matricesFile  ->  String with the full path of .mat file containing
%                       stiffness matrices and interpolation coefficients
%     potentialFilesFolder  ->  String with the full path of the folder containing
%                               the files that store potential values
%                           There are 2 options, as those files can be:
%                               1.- .MAT files previously created by the function 'prepareBinFilesForTorsoPropagation'
%                                   from the .bin files resulting from an Elvira simulation
%                               2.- .ENS and .CASE files containing potential values ('..._ENS_Vn_0000***.ens', '..._ENS_Vn.case')
%                                   resulting from the post-processing of an Elvira simulation 
%     ecgNodeFile  ->  String with the full path of a .mat file containing the index
%                      of the nodes of torso model where extracellular potentials must
%                      be computed. That .mat file have to be previously created by the
%                      function 'generateEcgNodesFile'.
%     output_id  ->  String containing the name of the .mat that will be created as
%                    an output in the same folder containing the input 'matricesFile'
%     parallel  ->  Integer value specifying whether or not parallel computing must be
%                   used to compute the torso propagation
%                       0 :: Non parallel computing
%                       1 :: Parallel computing
%     solver    ->  Integer value specifying which solver must be used to compute
%                   the torso propagation
%                       0 :: Direct solver   (integer)
%                       1 :: PCG (preconditioned conjugate gradient)
%                       2 :: bicgstab (preconditioned stabilized bi-conjugate gradient)
%                       3 :: bicgstabl(preconditioned stabilized(l) bi-conjugate gradiente)
%     tol_pcg  ->  {OPTIONAL} Tolerance parameter for iterative solvers
%                       Default value = 1e-6  (if not specified as an input)
%     maxIter  ->  {OPTIONAL} Maximum number of iterations for iterative solvers
%                       Default value = 100   (if not specified as an input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMPORTANT REMARKS  ------------------------------------------------ %%%
%      1. This function is ready to be run both in Matlab session mode
%         and also in deployed mode (compiled by 'mcc')
%      2. The code assumes that nodes and elements are numbers from
%         1 to N with no jumps
%      3. Performed tests indicate that a good set of options for solver
%         and preconditioner are the following:
%                       solver:     PCG    ->   1
%               preconditioner:     ichol  ->   incomplete Cholesky decomposition
%                factorisation:     ict
%               drop tolerance:     0.005
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHORSHIP -------------------------------------------------------- %%%
%                  Author:     Jose Felix Rodriguez ?????
%           Creation date:     ??/??/????
%       Last modification:     02/08/2017  by  Alejandro D. Lopez Perez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  step3_runTorsoPropagation (matricesFile, potentialFilesFolder, ecgNodeFile, 'BSPM_', 1, 1, 1e-6, 500)
function step3_runTorsoPropagation(matricesFile, potentialFilesFolder, ecgNodeFile, output_id, outputPath, auxPath, parallel, solver, tol_pcg, maxIter)


addpath(auxPath)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CHECK INPUTS
% Check whether the function is running in deployed mode (compiled by 'mcc')
% or in Matlab session mode
%if isdeployed  % if deployed mode
    %parallel = str2double (parallel);
%end  % if Matlab session, just continue

% Check options associated to iterative solvers solver
%switch nargin
    %case 6
        %% Take default values
        %tol_pcg = 1e-6;
        %maxIter = 100;
    %case 7
        %% Take default values
        %maxIter = 100;
       % % Check running mode
       % if isdeployed  % if deployed mode
            %tol_pcg = str2double (tol_pcg);
        %end  % if Matlab session, just continue
    %case 8
        %% Check running mode
        %if isdeployed  % if deployed mode
            %tol_pcg = str2double (tol_pcg);
            %maxIter = str2double (maxIter);
        %end  % if Matlab session, just continue
    %otherwise
        %error ('\n\n  ERROR ::: runTorsoPropagation :::  %s \n', 'Improper number of INPUTs');
%end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% POTENTIAL DATA
tic
fprintf ('\n\t ->  Looking for files containing potential data ... \n');
%%% Check whether those files are .mat or .ens files
% First check .mat files
potFilesList = dir (fullfile (potentialFilesFolder, '*.mat'));
if ~isempty (potFilesList)   % if .mat files
    % ...
    potFilesType = 'mat';
else  % if there is no .mat file ...
    % Then check .ens files
    potFilesList = dir (fullfile (potentialFilesFolder, '*Vn*.ens'));
    if ~isempty (potFilesList)   % if .ens files
        % ...
        potFilesType = 'ens';        
    else    % if there is no .ens file either
        error ('\n\n ERROR  :::  There is neither .MAT nor .ENS file in the folder specified for potential data \n\n\t\t %s \n\n', potentialFilesFolder);
    end
end
fprintf ('\t\t\t *  File type found in specified folder: \t %s \n', potFilesType);
%%% Extract files names
filesNames = cell (length(potFilesList), 1);
[filesNames{:}] = deal (potFilesList.name);
clear potFilesList;
%%% Get time axis
fprintf ('\t\t\t *  Reading file containing time axis ... \n');
% Check potential file type
switch potFilesType
    % if .MAT files
    case 'mat'
        % Look for 'time.mat' file
        indTimeFile = strcmp (filesNames, 'time.mat');
        % Load 'time' file
        time = load (fullfile (potentialFilesFolder, filesNames{indTimeFile}));
        varName = fieldnames (time);
        time = time.(varName{1});
        % Remove 'time' file from 'filesNames'
        filesNames(indTimeFile) = [];
    % if .ENS files
    case 'ens'
        % Look for .case files in specified folder
        caseFiles = dir (fullfile (potentialFilesFolder, '*_Vn.case'));
        % Check number of .case files found
        switch length(caseFiles)
            % if no .case file is found
            case 0
                error ('\n\n ERROR  :::  There is no .CASE file in the folder specified for potential data. It is necessary to get TIME AXIS \n\n\t\t %s \n\n', potentialFilesFolder);
            % if only one, as expected
            case 1
                % Open the file
                fid = fopen (fullfile (potentialFilesFolder, caseFiles.name), 'r');
                % Look for the time axis section into the .case file
                while ~feof(fid)
                    if ~isempty (strfind(fgetl(fid),'time values'))
                        break;
                    end
                end
                % Get time axis values
                time = cell2mat (textscan (fid, '%f', Inf));   %%% faster than using 'fscanf'
% % %                 time = fscanf (fid, '%f', Inf);
                % Close file
                fclose (fid);
            % if more than one .case file is found
            otherwise
                error ('\n\n ERROR  :::  There is more than one .CASE file in the folder specified for potential data. It is unclear which of them must be read in order get TIME AXIS \n\n\t\t %s \n\n', potentialFilesFolder);
        end
end
% Check whether the number of .mat files containing potential data matches
% the number of time steps
if length(filesNames) ~= length(time)  % if NOT
    error ('\n\n\t\t ERROR ::: ... \n\n\t\t\t The number of ''.%s'' files containing potential data must match the number of time steps in ''time'' file \n\n\n', potFilesType);
end  % if so, just continue
fprintf ('\t\t\t *  Elapsed time:   %s \n', executionTime(toc));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD STIFFNESS MATRICES
tic
fprintf ('\n\t ->  Loading .mat file containing stiffness matrices and interpolation coefficients ... \n');
% Load stiffness matrices to compute torso propagation
matrices = load (matricesFile);
%%% The following piece of code seems to be silly and completely unnecessary.
%%% However, if it is not executed as it stands, all variables contained in just
%%% loaded .mat file ('matricesFile') will not be available in 'parfor' loop in
%%% the case of parallel computing is required.
% Extract variables from structure 'matrices'
solver_type = matrices.solver_type;
Kright = matrices.Kright;
M = matrices.M;
b_length = matrices.b_length;
interpol = matrices.interpol;
nd_myo = matrices.nd_myo;
% Check type of solver to continue extracting all variables
switch matrices.solver_type
    % if Direct Solver
    case 0
        invK = matrices.invK;
        K = [];
        M1 = [];
        M2 = [];
    % if Iterative Solver
    otherwise
        K = matrices.K;
        M1 = matrices.M1;
        M2 = matrices.M2;
        invK = [];
end
clear matrices;   % Remove 'matrices' to free memory
fprintf ('\t\t\t *  Elapsed time:   %s \n', executionTime(toc));


%%%%%%%%%%%%%%%%%%%%%
%%% LOAD ECG NODES
tic
fprintf ('\n\t ->  Loading .mat file containing ECG nodes of torso model ... \n');
% Load ECG nodes
ecgNodes = load (ecgNodeFile);
varName = fieldnames (ecgNodes);
ecgNodes = ecgNodes.(varName{1});
fprintf ('\t\t\t *  Elapsed time:   %s \n', executionTime(toc));


%%%%%%%%%%%%%%%%%%%%%
%%% SOLVER
% Check whether 'solver' was specified as an input
if exist ('solver', 'var')  % if so
    % Save specified option in 'solver_type'
    if isdeployed  % if running in deployed mode
        solver_type = str2double (solver);
    else  % if Matlab session
        solver_type = solver;
    end
end  % if NOT, use the option definied in variable 'solver_type' stored in 'matricesFile'


%%%%%%%%%%%%%%%%%%%%%
%%% EXTRACELLULAR POTENTIALS COMPUTATION
tstart = tic;
% Initialise some variables
iter = zeros (length(time), 1);
flag = zeros (length(time), 1);
Vk = cell (1, length(time));
% Extract interpolation ID's and coefficients from 'interpol' structure
interpIDs = [interpol.Id]';
interpCoeffs = reshape ([interpol.alpha], length(interpol(1).alpha), length(interpol))';

% Check whether parallel computing is required
switch parallel
    %%%  NON PARALLEL COMPUTING  %%%
    case 0
        fprintf ('\n\t ->  Computing extracellular potentials ... \n');
        % Initialise previous state (required for Iterative Solvers) as an empty matrix for the first iteration
        Vprev = [];
        % Compute extracellular potentials for every time step
        for i = 1:length(time)
            tic
            % Load file containing potential data for all nodes of heart model at time 'time(i)'
            V = readPotentialFile (fullfile (potentialFilesFolder, filesNames{i}));
            % Interpolates potential from HEART MODEL to TORSO MODEL
            Vaux = V (interpIDs);
            clear V;    % Remove variables to free memory
            Vint = dot (interpCoeffs, Vaux, 2);
            % Remove variables to free memory
            clear Vaux;
            % ...
            b = zeros (b_length, 1);
            b(nd_myo) = -Kright * Vint;
            clear Vint;   % Remove variables to free memory

            % Compute extracellular potentials using specified solver
            % Check solver type
            if solver_type == 0   % Direct solver (muy caro desde el punto de vista de memoria)
                Vk{i} = invK * b;
            else  % Iterative solver
                % Potentials computation ...
                switch solver_type  % ... using specified iterative solver
                    case 1      % PCG (preconditioned conjugate gradient)
                        [Vk{i}, flag(i), ~, iter(i)] = pcg (K, b, tol_pcg, maxIter, M1, M2, Vprev);
                    case 2      % bicgstab (preconditioned stabilized bi-conjugate gradient)
                        [Vk{i}, flag(i), ~, iter(i)] = bicgstab (K, b, tol_pcg, maxIter, M1, M2, Vprev);
                    case 3      % bicgstabl(preconditioned stabilized(l) bi-conjugate gradiente)
                        [Vk{i}, flag(i), ~, iter(i)] = bicgstabl (K, b, tol_pcg, maxIter, M1, M2, Vprev);
                end
            end

            % Substract to Vk its average in order to guarantee the solution is zero mean          
	    Vk{i} = Vk{i} - dot (M, Vk{i});
            % Check solver type once again
            if solver_type ~= 0  % if Iterative solver
                % Save result as previous state for next iteration
                %%% Passing the previous state to any iterative solver as an initial state is supposed
                %%% to provide a much faster convergence of the solution to the desired tolerance
                Vprev = Vk{i};
            end  % if Direct solver, just continue
            % Take from the result only those values corresponding to ECG nodes
            Vk{i} = Vk{i}(ecgNodes);
            fprintf ('\t\t\t *  Computed torso propagation for time point #%d (t = %.2f) from a total of %d:   %s \n', i, time(i), length(time), executionTime(toc));
        end
        %%% END of Non Parallel Computing %%%

    %%%  PARALLEL COMPUTING  %%%
    case 1
        %%%%%%%%%%%%%
        %%% OPEN PARALLEL POOL
        tic
        fprintf ('\n\t ->  Preparing parallel computing ... \n');
        % Check whether parallel pool is already open
        if ~isempty (gcp ('nocreate'))  % if so
            % Close it
            evalc ('delete (gcp);');
        end  % if NOT, just continue
        % Check OS of the system and get the number of processors available in machine
        if isunix  % if UNIX-based system
            [~, numCPUs] = system ('nproc');         % for LINUX/UNIX systems
        elseif ispc   % if Windows running on PC
            numCPUs = getenv ('NUMBER_OF_PROCESSORS');  % for WINDOWS systems
        end
        fprintf ('\t\t\t *  Number of cores available for parallel computing:  %s \n', numCPUs);
        numCPUs = str2double (numCPUs);
        % Set 'numCPUs' as the maximum allowed number of workers in 'local' profile for parallel computing
        parOptions = parcluster ('local');
        parOptions.NumWorkers = numCPUs;
        % Open the parallel pool
        msg = evalc ('pool = parpool (''local'', numCPUs);');
        pool.IdleTimeout = Inf;   % Set 'No automatic shut down'
        fprintf ('\t\t\t *  %s \n', msg(1:end-2));
        % Execution time
        fprintf ('\t\t\t *  Elapsed time for opening parallel pool:   %s \n', executionTime(toc))
        %%%%%%%%%%%%%%%%%%
        
        fprintf ('\n\t ->  Computing extracellular potentials ...\n');

        %%%%%%%%%%%%%
        % Compute extracellular potentials for every time step
        parfor i = 1:length(time)
            % Load file containing potential data for all nodes of heart model at time 'time(i)'
            V = readPotentialFile (fullfile (potentialFilesFolder, filesNames{i}));            
            % Interpolate potential from HEART MODEL to TORSO MODEL
            Vaux = V (interpIDs);
            V = [];  %#ok<NASGU>  %%% Empty variables to free memory (the use of 'clear' is not allowed in 'parfor' loops)
            Vint = dot (interpCoeffs, Vaux, 2);
            Vaux = [];   %#ok<NASGU>    %%% Empty variables to free memory (the use of 'clear' is not allowed in 'parfor' loops)
            % ...
            b = zeros (b_length, 1);
            b(nd_myo) = -Kright * Vint;
            Vint = [];  %#ok<NASGU>    %%% Empty variables to free memory (the use of 'clear' is not allowed in 'parfor' loops)
            
            % Compute extracellular potentials using specified solver
            switch solver_type
                case 0      % Direct solver (muy caro desde el punto de vista de memoria)
                    Vk{i} = invK * b;
                case 1      % PCG (preconditioned conjugate gradient)
                    [Vk{i}, flag(i), ~, iter(i)] = pcg (K, b, tol_pcg, maxIter, M1, M2, []); %#ok<PFOUS>
                case 2      % bicgstab (preconditioned stabilized bi-conjugate gradient)
                    [Vk{i}, flag(i), ~, iter(i)] = bicgstab (K, b, tol_pcg, maxIter, M1, M2, []);
                case 3      % bicgstabl(preconditioned stabilized(l) bi-conjugate gradiente)
                    [Vk{i}, flag(i), ~, iter(i)] = bicgstabl (K, b, tol_pcg, maxIter, M1, M2, []);
            end
            % Substract to Vk its average in order to guarantee that the solution is zero mean
            Vk{i} = Vk{i} - dot (M, Vk{i});
            % Take from the result only those values corresponding to ECG nodes
            Vk{i} = Vk{i}(ecgNodes);
        end
        % Delete parallel pool
        evalc ('delete (pool);');
        %%% END of Parallel Computing %%%
end
%%% END of Extracellular Potentials Computation %%%

fprintf ('\n\t\t\t *  Computations FINISHED !!!!!! \n\n');
fprintf ('\t\t\t *  Average number of iterations per increment:  %d \n', round(mean(iter)));

% Save results
fprintf ('\t\t\t *  Saving results in a .mat file ... \n');

resultsFile = setNewFileName (outputPath, strcat(output_id, '.mat'));%sandra
V = cell2mat (Vk); %#ok<NASGU>  %%% Transform 'Vk' from cell array into a single matrx
save (resultsFile, 'time', 'V', 'ecgNodes', 'flag', 'iter', '-v7.3');
% Final message
fprintf ('\n\t ->  FINISH  :::  Elapsed time for whole process:   %s \n\n', executionTime(toc(tstart)));
% return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  AUXILIARY FUNCTIONS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function V = readPotentialFile (potFilePath)
% % Get the extension of potential file
% [~, ~, extension] = fileparts (potFilePath);
% % And check the type of file that contains potential data
% switch extension
%   % if .MAT files
%     case '.mat'
%         % Load .mat file
%         V = load (potFilePath);
%         varName = fieldnames (V);
%         V = V.(varName{1});
%   % if .ENS files
%     case '.ens'
%         % Open .ens file
%         fidPot = fopen (potFilePath, 'r');
%         % Look for the 'coordinates' section into the .ens file
%         while ~feof(fidPot)
%             if ~isempty (strfind(fgetl(fidPot), 'coordinates'))
%                 break;
%             end
%         end
%         % Get potential values
%         V = cell2mat (textscan (fidPot, '%f', Inf));   %%% faster than using 'fscanf'
% %%%%        V = fscanf (fidPot, '%f', Inf);
%         % Close .ens file
%         fclose (fidPot);
% end
end