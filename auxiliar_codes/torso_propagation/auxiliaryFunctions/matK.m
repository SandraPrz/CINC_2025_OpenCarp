%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%  Element matrices
function [Ke, Ve, Me] = matK (De, xe)
% De: conductivity tensor
% xe: nodal coordinates
% Derivative shape functions in natural coordinates
dN = [-1 1 0 0;
      -1 0 1 0;
      -1 0 0 1];
% Derivative shape functions in cartesian coordinates
% Jacobian
Jac = dN * xe;
% % % Jac = zeros(3);
% % % Jac(:,1) = dN * xe(:,1);
% % % Jac(:,2) = dN * xe(:,2);
% % % Jac(:,3) = dN * xe(:,3);
% Jacobian determinant
detJac = det(Jac);
% Volume of the element
Ve = detJac/6.0;
if (Ve <= 0.0)
    fprintf (' WARNING !!!!  Element jacobian less than or equal to zero:   %f \n', Ve);
    return;
end
B = Jac\dN;
wgp = 1/6;
Ke = wgp*detJac*B'*(De*B);
Me = 0.25*Ve*ones(4,1);
return