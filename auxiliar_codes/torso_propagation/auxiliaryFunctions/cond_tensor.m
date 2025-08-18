%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Conductivity tensor
function De = cond_tensor (sig0, r_sig, nfib)
% Conductivity tensor
if (sig0 <= 1.0e-3)
    sig0 = 0.001;
end
if (r_sig > 0)
    % Normalizing fiber direction
    norm_nfib = norm(nfib);
    if (norm_nfib <= 1.0e-6)
        nfib = zeros(1,3);
        r_sig = 1.0;
    else
        nfib = nfib/norm_nfib;
    end
    De = sig0*((1-r_sig)*nfib'*nfib + r_sig*eye(3)); %#ok<MHERM>
else
    De = sig0*eye(3);
end
return