function [L,H] = xdwtfun(data,wname)
matSize = size(data);
for i = 1:matSize(2)
    for j = 1:matSize(3)
        line = data(:,i,j);
        [cL,cH]=dwt(line,wname);
        L(:,i,j)=cL;
        H(:,i,j)=cH;
    end
end
end