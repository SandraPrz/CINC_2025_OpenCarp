%%% --------------------------------------
% Calculates interpolating coefficients based on radial function principle
function alpha = inter_coef (dist)
ncoef = length(dist);
dist = dist/mean(dist);
alpha = ones(ncoef,1);
for i = 1:ncoef
    for j = 1:ncoef
        if (j ~= i)
            alpha(i) = alpha(i) * dist(j);
        end
    end
end
denominator = sum(alpha);
alpha = alpha/denominator;
return