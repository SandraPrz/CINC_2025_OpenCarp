%%% -----------------------------------------------
% This subroutine sets the sparse structures for the stiffness matrices
function [IIK, IJK, AxK, IIKr, IJKr, AxKr] = sparse_setting (torsoElements, torsoNodes)
%%% WHOLE TORSO
fprintf ('\n\t\t\t *  Processing TORSO MODEL. Dimensioning sparse structures for stiffness matrices ... \n');
[IIK, IJK, AxK] = sparse_dim (torsoNodes.NumNodes, torsoElements.NumElements, torsoElements.ConnectivityList);
%%% MYOCARDIAL ELEMENTS
fprintf ('\n\t\t\t *  Processing MYOCARDIAL ELEMENTS. Dimensioning sparse structures for stiffness matrices ...\n');
elm = torsoElements.ConnectivityList (torsoElements.MyoElements, :);
for k = 1:4
    elm(:,k) = torsoNodes.MyoNodes (elm(:,k));
end
[IIKr, IJKr, AxKr] = sparse_dim (torsoNodes.NumMyoNodes, torsoElements.NumMyoElements, elm);
return

%%% -----------------------------------------------
% This function dimensions the sparse structures for the stiffness matrices
function [II, IJ, Ax] = sparse_dim (nnod, nelm, elm)
tic
fprintf ('\t\t\t\t\t -  Building neighbor ELEMENTS to NODE graph ... \n');
IIe = zeros(nnod+1,1);
for i = 1:nelm
    cone = elm (i,:);
    IIe(cone+1) = IIe(cone+1) + 1;
end
Id = find (IIe(2:end) == 0);
if(length(Id)>0) %#ok<*ISMT>
    fprintf ('\t\t\t\t\t\t\t +  There are unconnected nodes in the model ... \n');
    fprintf ('\t\t\t\t\t\t\t\t\t\t %d \n', Id);
end
IIe(1) = 1; % displacement vector
max_vec = 0;
min_vec = 1000;
for i = 2:(nnod+1)
    if (IIe(i) >= max_vec)
        max_vec = IIe(i);
    end
    if ((IIe(i)<=min_vec) && (IIe(i)>0))
        min_vec = IIe(i);
    end
    IIe(i) = IIe(i) + IIe(i-1);
end
Nneigh_tot = IIe(nnod+1);
Nneigh = zeros(nnod,1);
IJe = zeros(Nneigh_tot,1);
for i = 1:nelm
    cone = elm (i,:);
    for j = 1:length(cone)
        nnum = cone(j);
        IJe(IIe(nnum)+Nneigh(nnum)) = i;
    end
    Nneigh(cone) = Nneigh(cone) + 1;
end
fprintf ('\t\t\t\t\t\t\t +  Maximum number of neighbor elements to a node:   %d \n', max_vec);
fprintf ('\t\t\t\t\t\t\t    Minimum number of neighbor elements to a node:   %d \n', min_vec);
fprintf ('\t\t\t\t\t\t\t +  Elapsed time:   %s \n', executionTime(toc));

tic
fprintf ('\t\t\t\t\t -  Building neighbor NODE to NODE graph ...\n');
% Computing number of nonzero elements in K
II = zeros (nnod+1, 1);
for i = 1:nnod
    cone_neigh = elm (IJe(IIe(i):(IIe(i+1)-1)), :);
    nnz_row = length(unique(cone_neigh(:)));
    II(i+1) = nnz_row;
end
II(1) = 1; % Displacement vector
max_vec = 0;
min_vec = 1000;
for i = 2:(nnod+1)
    if(II(i) >= max_vec)
        max_vec = II(i);
    end
    if (II(i) <= min_vec)
        min_vec = II(i);
    end
    II(i) = II(i) + II(i-1);
end
fprintf ('\t\t\t\t\t\t\t +  Number of nonzero elements in K:   %d \n', II(nnod+1)-1);
fprintf ('\t\t\t\t\t\t\t                     Bandwidth of K:   %d \n', max_vec);
IJ = zeros(II(nnod+1)-1,1);
Ax = IJ;
for i = 1:nnod
    cone_neigh = elm (IJe(IIe(i):(IIe(i+1)-1)), :);
    cone_neigh = unique (cone_neigh);
    IJ(II(i):(II(i+1)-1)) = cone_neigh;
end
fprintf ('\t\t\t\t\t\t\t +  Elapsed time:   %s \n', executionTime(toc));
return