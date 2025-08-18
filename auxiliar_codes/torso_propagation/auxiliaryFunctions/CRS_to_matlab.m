function IInew = CRS_to_matlab (II)
IInew = zeros(II(end)-1,1);
n = length(II)-1;
for i = 1:n
    IInew(II(i):(II(i+1)-1)) = i;
end
return