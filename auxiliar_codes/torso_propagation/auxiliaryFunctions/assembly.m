%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Assembly subroutine
function [K, M, Kright, Vtotal] = assembly (torsoElements, torsoNodes, solver_type, lambda)
% Dimensioning sparse structures for K and Kright
[IIK, IJK, AxK, IIKr, IJKr, AxKr] = sparse_setting (torsoElements, torsoNodes);
tstart = tic;
fprintf ('\n\t\t\t *  Generating stiffness matrices. It could take several minutes ... \n');
% Create variables to give some feedback during the process
step = 10;   % specified in percentage
feedback = step:step:100;
feedback (2,:) = round (linspace (round(step*double(torsoElements.NumElements)/100), double(torsoElements.NumElements), length(feedback)));
nextFeedback = 1;   % counter
% Initialise some variables
npe = size (torsoElements.ConnectivityList, 2);   % Number of nodes per element (tetrahedra -> 4 nodes per element)
Vtotal = 0.0;
M = zeros (torsoNodes.NumNodes, 1);
% Go through all elements of torso model
for i = 1:torsoElements.NumElements
    cone = torsoElements.ConnectivityList(i,:);
    xe = torsoNodes.Coordinates(cone,:);
    De = cond_tensor (torsoElements.LongCV(i), torsoElements.AnisotropyRatio(i), torsoElements.FibreOrientation(i,:));
    % Element matrix
    [Ke, Ve, Me] = matK (De, xe);
    Vtotal = Vtotal + Ve;
    M(cone) = M(cone) + Me;
    for j = 1:npe
        idi = IIK(cone(j));
        for k = 1:npe
            idj = cone(k);
            cont = 0;
            while (IJK(idi+cont)~=idj)
                cont = cont + 1;
            end
            AxK(idi+cont) = AxK(idi+cont) + Ke(j,k);
        end
    end
    
    % Check whether current cell/element belongs to 'myocardium'
    if torsoElements.MyoElements(i)
        cone_r = torsoNodes.MyoNodes(cone);
        for j = 1:npe
            idi = IIKr(cone_r(j));
            for k = 1:npe
                idj = cone_r(k);
                cont = 0;
                while (IJKr(idi+cont)~=idj)
                    cont = cont + 1;
                end
                AxKr(idi+cont) = AxKr(idi+cont) + Ke(j,k)/(1+lambda);
            end
        end
    end  % if NOT, just continue
    % Give some feedbak about the progress of the process
    if i == feedback(2,nextFeedback)
        fprintf ('\t\t\t\t\t %3d%% accomplished.  Elapsed time:  %s \n', feedback(1,nextFeedback), executionTime(toc(tstart)));
        nextFeedback = nextFeedback + 1;   % update counter for next comparison
    end
end
IIK = CRS_to_matlab (IIK);
IIKr = CRS_to_matlab (IIKr);
M = sparse (M);
K = sparse (IIK, IJK, AxK, double(torsoNodes.NumNodes), double(torsoNodes.NumNodes));
Kright = sparse (IIKr, IJKr, AxKr, double(torsoNodes.NumMyoNodes), double(torsoNodes.NumMyoNodes));
M = M/Vtotal;
% if Direct solver
if (solver_type == 0)
    K = [K M; M' 0];
end
return