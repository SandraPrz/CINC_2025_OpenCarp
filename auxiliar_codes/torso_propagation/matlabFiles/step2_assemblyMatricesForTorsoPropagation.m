%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   step2_assemblyMatricesForTorsoPropagation (dataFile, solver_type, prec_string, type_fact, droptol)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%%%         2nd STEP IN TORSO PROPAGATION PROCESS         %%%%%%%%%%
%%%%%%%%%%%%%% --------------------------------------------- %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  'step2_assemblyMatricesForTorsoPropagation' generates the stiffnes matrices
%  that will be later required to compute the torso propagation by means
%  of the function 'runTorsoPropagation'. Those matrices will be stored
%  in a new .mat created in the same folder containing the input 'dataFile'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTs ------------------------------------------------------------ %%%
%     dataFile     ->  String with the full path of the .mat file, previously
%                      created by the function 'generateFilesForTorsoPropagation',
%                      containing all the data related to torso model
%     solver_type  ->  Integer value specifying which solver must be used
%                      to compute the torso propagation
%                           0 :: Direct solver
%                           1 :: PCG (preconditioned conjugate gradient)
%                           2 :: bicgstab (preconditioned stabilized bi-conjugate gradient)
%                           3 :: bicgstabl(preconditioned stabilized(l) bi-conjugate gradiente)
%     prec_string  ->  String defining the type of preconditioner to be used
%                             'ilu' :: incomplete LU
%                           'ichol' :: incomplete Cholesky decomposition
%     type_fact    ->  String defining the type of factorization to be used
%                           for ILU :: 'nofill', 'ilutp', 'crout'
%                         for ICHOL :: 'nofill', 'ict'
%     droptol      ->  Float value corresponding to the 'dropoff tolerance'
%                      from factorization. Not used with 'nofill'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMPORTANT REMARKS  ------------------------------------------------ %%%
%      1. This function is ready to be run both in Matlab session mode and
%         also in deployed mode (compiled by 'mcc')
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
%       Last modification:     10/03/2017  by  Alejandro D. Lopez Perez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  step2_assemblyMatricesForTorsoPropagation (dataFile, 1, 'ichol', 'ict', 0.005)
%function step2_assemblyMatricesForTorsoPropagation (dataFile, solver_type, prec_string, type_fact, droptol)


dataFile= '/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/PSM1_PSM4/PSM1_ctrl/post_S2/torso/Torso/torsoPropagation_allNormal/PSM1_torso_organLabel_dataForTorsoPropagation.mat'
solver_type=1
prec_string='ichol'
type_fact='ict'
droptol=0.005
addpath('/gpfs/projects/upv100/WORK-SANDRA/SANDRA_TFM/drugs/auxiliaryFunctions_RIGEL/')



% This constant ('lambda') is the ratio between the intracellular and
% extracellular conduction tensors according to the monodomain approach
lambda = 3.64;  % monodomain

% Get folder path
[dataFolder, ~, ~] = fileparts (dataFile);

% Check running mode
if isdeployed  % if running in deployed mode (comiled by 'mcc')
    % Manage numeric inputs
    solver_type = str2double (solver_type);
    droptol = str2double (droptol);
else   % if Matlab session
    % Create a log file
    diary (fullfile (dataFolder, 'assemblyMatrices_LOG.txt'));
end

solv_str = cell (4,1);
solv_str{1} = 'dir';
solv_str{2} = 'pcg';
solv_str{3} = 'bicgstab';
solv_str{4} = 'bicgstabl';

tstart_glob = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  READ TORSO MODEL DATA  %%%
tstart = tic;
fprintf ('\n\t ->  Reading TORSO MODEL data ... \n');
fprintf ('\t\t\t *  Loading .mat file ... \n');
load (dataFile);

% Reordering myocardial nodes for right hand side assemblying
nd_myo = find (torsoNodes.MyoNodes); %#ok<NODEF>
% Renumber myocardial nodes
torsoNodes.MyoNodes = torsoNodes.MyoNodes .* cumsum (torsoNodes.MyoNodes);
fprintf ('\t\t\t *  Done reading and renumberign \n');
fprintf ('\t\t\t *  Elapsed time:   %s \n', executionTime(toc(tstart)));

% Assembling matrices
tstart = tic;
fprintf ('\n\t ->  Assembling matrices ... \n');
[K, M, Kright, ~] = assembly (torsoElements, torsoNodes, solver_type, lambda);
fprintf ('\t\t\t *  Matrices assembled \n');
fprintf ('\t\t\t *  Elapsed time:   %s \n', executionTime(toc(tstart)));

% Preparing for the solution of the system
tstart = tic;
fprintf ('\n\t ->  Initiating solver preprocessing ... \n');

% Check specified solver
switch solver_type
    %%% Direct solver (muy caro desde el punto de vista de memoria)
    case 0
        fprintf ('\t\t\t *  Using DIRECT SOLVER \n');
        fprintf ('\t\t\t\t\t -  Regularization ... \n');
        % at this point we impose the constraint that mean(Vo)=0. This condition
        % is equivalent to M*Vo = 0 donde M es la matriz de masa normalizada por
        % el volumen total del torso
        fprintf ('\t\t\t\t\t -  Inverting ... \n');
        Imat = sparse (1:(torsoNodes.NumNodes+1), 1:(torsoNodes.NumNodes+1), ones(torsoNodes.NumNodes+1,1));    %%%  Imat = sparse (1:(nnod_trg+1), 1:(nnod_trg+1), ones(nnod_trg+1,1));
        invK = K\Imat;
        clear K Imat;
        b_length = torsoNodes.NumNodes + 1;
        
% % %         % Generate transfer matrix 'A'
% % %         tic
% % %         fprintf ('\t\t\t\t\t -  Generating transfer matrix ... \n');
% % %         A = zeros (nnod_myo, nnod_trg);
% % %         for i = 1:nnod_myo
% % %             A (i,interpol(i).Id) = interpol(i).alpha;
% % %         end
% % %         A = -Kright * A;
% % %         invKaux = invK (nd_ecg, :);
% % %         Amatrix = invKaux (:, nd_myo) * A;
        
        %%% Save matrices needed to compute torso propagation
        fprintf ('\t\t\t *  Saving matrices in a .mat file \n');
        matricesFile = setNewFileName (dataFolder, 'matricesForDirectSolver.mat');
        save (matricesFile, 'solver_type', 'Kright', 'M', 'invK', 'b_length', 'interpol', 'nd_myo');  %%%% , '-v7.3');
% % %         save (matricesFile, 'solver_type', 'Kright', 'M', 'invK', 'b_length', 'interpol', 'nd_myo', 'nd_ecg', 'Amatrix');
        fprintf ('\t\t\t *  Done. Elapsed time:   %s \n', executionTime(toc(tstart)));
    
    %%% Iterative solver
    case {1,2,3}
        fprintf ('\t\t\t *  Using ITERATIVE SOLVER:  ''%s'' \n', solv_str{solver_type+1});
        fprintf ('\t\t\t\t\t -  Computing preconditioner:  ''%s'' \n', prec_string);
        switch prec_string
            case 'ilu'
                switch type_fact
                    case 'nofill'
                        setup.type = type_fact;
                    case 'ilutp'
                        setup.type = type_fact;
                    case 'crout'
                        setup.type = type_fact;
                    otherwise
                        fprintf ('\t\t\t\t\t -  Specified factorization not defined. Switching to nofill \n')
                        setup.type = 'nofill';
                end
                setup.droptol = droptol;
                [M1, M2] = ilu (K,setup); %#ok<*NASGU,*ASGLU>
            case 'ichol'
                switch type_fact
                    case 'nofill'
                        setup.type = type_fact;
                    case 'ict'
                        setup.type = type_fact;
                    otherwise
                        fprintf ('\t\t\t\t\t -  Specified factorization not defined. Switching to nofill \n')
                        setup.type = 'nofill';
                end
                setup.droptol = droptol;
                M1 = ichol (K, setup);
                M2 = M1';
        end
        fprintf ('\t\t\t\t\t -  Factorization : ''%s'' \n', type_fact);
        fprintf ('\t\t\t\t\t -  Drop tolerance: %e (not used with nofill) \n', droptol);
        b_length = torsoNodes.NumNodes;
        
        %%% Save matrices needed to compute torso propagation
        fprintf ('\t\t\t *  Saving matrices in a .mat file \n');
        matricesFile = setNewFileName (dataFolder, 'matricesForIterativeSolver.mat');
        save (matricesFile, 'solver_type', 'K', 'Kright', 'M', 'M1', 'M2', 'b_length', 'interpol', 'nd_myo', 'prec_string', 'type_fact'); %%% , '-v7.3');
        fprintf ('\t\t\t *  Finished solver preprocessing. Elapsed time:   %s \n', executionTime(toc(tstart)));
end

% Final message
fprintf ('\n\t ->  FINISH.  Elapsed time for whole process:   %s \n\n', executionTime(toc(tstart_glob)))

% Check running mode
if ~isdeployed  % if Matlab session
    % Close log file
    diary;
    diary off;
end
% return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Auxiliary Sub-routines
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% Assembly subroutine
% function [K, M, Kright, Vtotal] = assembly (torsoElements, torsoNodes, solver_type, lambda)
% % Dimensioning sparse structures for K and Kright
% [IIK, IJK, AxK, IIKr, IJKr, AxKr] = sparse_setting (torsoElements, torsoNodes);
% tstart = tic;
% fprintf ('\n\t\t\t *  Generating stiffness matrices. It could take several minutes ... \n');
% % Create variables to give some feedback during the process
% step = 10;   % specified in percentage
% feedback = step:step:100;
% feedback (2,:) = round (linspace (round(step*double(torsoElements.NumElements)/100), double(torsoElements.NumElements), length(feedback)));
% nextFeedback = 1;   % counter
% % Initialise some variables
% npe = size (torsoElements.ConnectivityList, 2);   % Number of nodes per element (tetrahedra -> 4 nodes per element)
% Vtotal = 0.0;
% M = zeros (torsoNodes.NumNodes, 1);
% % Go through all elements of torso model
% for i = 1:torsoElements.NumElements
%     cone = torsoElements.ConnectivityList(i,:);
%     xe = torsoNodes.Coordinates(cone,:);
%     De = cond_tensor (torsoElements.LongCV(i), torsoElements.AnisotropyRatio(i), torsoElements.FibreOrientation(i,:));
%     % Element matrix
%     [Ke, Ve, Me] = matK (De, xe);
%     Vtotal = Vtotal + Ve;
%     M(cone) = M(cone) + Me;
%     for j = 1:npe
%         idi = IIK(cone(j));
%         for k = 1:npe
%             idj = cone(k);
%             cont = 0;
%             while (IJK(idi+cont)~=idj)
%                 cont = cont + 1;
%             end
%             AxK(idi+cont) = AxK(idi+cont) + Ke(j,k);
%         end
%     end
%     
%     % Check whether current cell/element belongs to 'myocardium'
%     if torsoElements.MyoElements(i)
%         cone_r = torsoNodes.MyoNodes(cone);
%         for j = 1:npe
%             idi = IIKr(cone_r(j));
%             for k = 1:npe
%                 idj = cone_r(k);
%                 cont = 0;
%                 while (IJKr(idi+cont)~=idj)
%                     cont = cont + 1;
%                 end
%                 AxKr(idi+cont) = AxKr(idi+cont) + Ke(j,k)/(1+lambda);
%             end
%         end
%     end  % if NOT, just continue
%     % Give some feedbak about the progress of the process
%     if i == feedback(2,nextFeedback)
%         fprintf ('\t\t\t\t\t %3d%% accomplished.  Elapsed time:  %s \n', feedback(1,nextFeedback), executionTime(toc(tstart)));
%         nextFeedback = nextFeedback + 1;   % update counter for next comparison
%     end
% end
% IIK = CRS_to_matlab (IIK);
% IIKr = CRS_to_matlab (IIKr);
% M = sparse (M);
% K = sparse (IIK, IJK, AxK, double(torsoNodes.NumNodes), double(torsoNodes.NumNodes));
% Kright = sparse (IIKr, IJKr, AxKr, double(torsoNodes.NumMyoNodes), double(torsoNodes.NumMyoNodes));
% M = M/Vtotal;
% % if Direct solver
% if (solver_type == 0)
%     K = [K M; M' 0];
% end
% return

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%  Conductivity tensor
% function De = cond_tensor (sig0, r_sig, nfib)
% % Conductivity tensor
% if (sig0 <= 1.0e-3)
%     sig0 = 0.001;
% end
% if (r_sig > 0)
%     % Normalizing fiber direction
%     norm_nfib = norm(nfib);
%     if (norm_nfib <= 1.0e-6)
%         nfib = zeros(1,3);
%         r_sig = 1.0;
%     else
%         nfib = nfib/norm_nfib;
%     end
%     De = sig0*((1-r_sig)*nfib'*nfib + r_sig*eye(3)); %#ok<MHERM>
% else
%     De = sig0*eye(3);
% end
% return

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%  Element matrices
% function [Ke, Ve, Me] = matK (De, xe)
% % De: conductivity tensor
% % xe: nodal coordinates
% % Derivative shape functions in natural coordinates
% dN = [-1 1 0 0;
%       -1 0 1 0;
%       -1 0 0 1];
% % Derivative shape functions in cartesian coordinates
% % Jacobian
% Jac = dN * xe;
% % % % Jac = zeros(3);
% % % % Jac(:,1) = dN * xe(:,1);
% % % % Jac(:,2) = dN * xe(:,2);
% % % % Jac(:,3) = dN * xe(:,3);
% % Jacobian determinant
% detJac = det(Jac);
% % Volume of the element
% Ve = detJac/6.0;
% if (Ve <= 0.0)
%     fprintf (' WARNING !!!!  Element jacobian less than or equal to zero:   %f \n', Ve);
%     return;
% end
% B = Jac\dN;
% wgp = 1/6;
% Ke = wgp*detJac*B'*(De*B);
% Me = 0.25*Ve*ones(4,1);
% return

% %%% -----------------------------------------------
% % This subroutine sets the sparse structures for the stiffness matrices
% function [IIK, IJK, AxK, IIKr, IJKr, AxKr] = sparse_setting (torsoElements, torsoNodes)
% %%% WHOLE TORSO
% fprintf ('\n\t\t\t *  Processing TORSO MODEL. Dimensioning sparse structures for stiffness matrices ... \n');
% [IIK, IJK, AxK] = sparse_dim (torsoNodes.NumNodes, torsoElements.NumElements, torsoElements.ConnectivityList);
% %%% MYOCARDIAL ELEMENTS
% fprintf ('\n\t\t\t *  Processing MYOCARDIAL ELEMENTS. Dimensioning sparse structures for stiffness matrices ...\n');
% elm = torsoElements.ConnectivityList (torsoElements.MyoElements, :);
% for k = 1:4
%     elm(:,k) = torsoNodes.MyoNodes (elm(:,k));
% end
% [IIKr, IJKr, AxKr] = sparse_dim (torsoNodes.NumMyoNodes, torsoElements.NumMyoElements, elm);
% return
% 
% %%% -----------------------------------------------
% % This function dimensions the sparse structures for the stiffness matrices
% function [II, IJ, Ax] = sparse_dim (nnod, nelm, elm)
% tic
% fprintf ('\t\t\t\t\t -  Building neighbor ELEMENTS to NODE graph ... \n');
% IIe = zeros(nnod+1,1);
% for i = 1:nelm
%     cone = elm (i,:);
%     IIe(cone+1) = IIe(cone+1) + 1;
% end
% Id = find (IIe(2:end) == 0);
% if(length(Id)>0) %#ok<*ISMT>
%     fprintf ('\t\t\t\t\t\t\t +  There are unconnected nodes in the model ... \n');
%     fprintf ('\t\t\t\t\t\t\t\t\t\t %d \n', Id);
% end
% IIe(1) = 1; % displacement vector
% max_vec = 0;
% min_vec = 1000;
% for i = 2:(nnod+1)
%     if (IIe(i) >= max_vec)
%         max_vec = IIe(i);
%     end
%     if ((IIe(i)<=min_vec) && (IIe(i)>0))
%         min_vec = IIe(i);
%     end
%     IIe(i) = IIe(i) + IIe(i-1);
% end
% Nneigh_tot = IIe(nnod+1);
% Nneigh = zeros(nnod,1);
% IJe = zeros(Nneigh_tot,1);
% for i = 1:nelm
%     cone = elm (i,:);
%     for j = 1:length(cone)
%         nnum = cone(j);
%         IJe(IIe(nnum)+Nneigh(nnum)) = i;
%     end
%     Nneigh(cone) = Nneigh(cone) + 1;
% end
% fprintf ('\t\t\t\t\t\t\t +  Maximum number of neighbor elements to a node:   %d \n', max_vec);
% fprintf ('\t\t\t\t\t\t\t    Minimum number of neighbor elements to a node:   %d \n', min_vec);
% fprintf ('\t\t\t\t\t\t\t +  Elapsed time:   %s \n', executionTime(toc));
% 
% tic
% fprintf ('\t\t\t\t\t -  Building neighbor NODE to NODE graph ...\n');
% % Computing number of nonzero elements in K
% II = zeros (nnod+1, 1);
% for i = 1:nnod
%     cone_neigh = elm (IJe(IIe(i):(IIe(i+1)-1)), :);
%     nnz_row = length(unique(cone_neigh(:)));
%     II(i+1) = nnz_row;
% end
% II(1) = 1; % Displacement vector
% max_vec = 0;
% min_vec = 1000;
% for i = 2:(nnod+1)
%     if(II(i) >= max_vec)
%         max_vec = II(i);
%     end
%     if (II(i) <= min_vec)
%         min_vec = II(i);
%     end
%     II(i) = II(i) + II(i-1);
% end
% fprintf ('\t\t\t\t\t\t\t +  Number of nonzero elements in K:   %d \n', II(nnod+1)-1);
% fprintf ('\t\t\t\t\t\t\t                     Bandwidth of K:   %d \n', max_vec);
% IJ = zeros(II(nnod+1)-1,1);
% Ax = IJ;
% for i = 1:nnod
%     cone_neigh = elm (IJe(IIe(i):(IIe(i+1)-1)), :);
%     cone_neigh = unique (cone_neigh);
%     IJ(II(i):(II(i+1)-1)) = cone_neigh;
% end
% fprintf ('\t\t\t\t\t\t\t +  Elapsed time:   %s \n', executionTime(toc));
% return

%%% -----------------------------------------------
% function IInew = CRS_to_matlab (II)
% IInew = zeros(II(end)-1,1);
% n = length(II)-1;
% for i = 1:n
%     IInew(II(i):(II(i+1)-1)) = i;
% end
% return