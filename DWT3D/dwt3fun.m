function [LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH]=dwt3fun(data,wname)
[L,H]=xdwtfun(data,wname);
[LL,LH]=ydwtfun(L,wname);
[HL,HH]=ydwtfun(H,wname);
[LLL,LLH]=zdwtfun(LL,wname);
[LHL,LHH]=zdwtfun(LH,wname);
[HLL,HLH]=zdwtfun(HL,wname);
[HHL,HHH]=zdwtfun(HH,wname);
end