function data=idwt3fun(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH,wname)
disp('1')
HH = zidwt(HHL,HHH,wname);
disp('1')
HL = zidwt(HLL,HLH,wname);
disp('1')
LH = zidwt(LHL,LHH,wname);
disp('1')
LL = zidwt(LLL,LLH,wname);
disp('2')
H = yidwt(HL,HH,wname);
disp('2')
L = yidwt(LL,LH,wname);
disp('3')
data = xidwt(L,H,wname);
end