function data = xidwt(L,H,wname)
Ls = size(L);
Hs = size(H);
flag = isequal(Ls,Hs);
if flag ~= 1
    disp('input matrix size not equal')
    return
end
matSize = size(L);
for i = 1:matSize(2)
    for j = 1:matSize(3)
        L1 = L(:,i,j);
        H1 = L(:,i,j);
        line = idwt(L1,H1,wname);
        data(:,i,j)=line;
    end
end
end