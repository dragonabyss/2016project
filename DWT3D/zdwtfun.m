function [L,H] = zdwtfun(data,wname)
matSize = size(data);
for i = 1:matSize(1)
    for j = 1:matSize(2)
        line = data(i,j,:);
        [cL,cH]=dwt(line,wname);
        L(i,j,:)=cL;
        H(i,j,:)=cH;
    end
end
end