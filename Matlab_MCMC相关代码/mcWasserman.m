% mcWasserman.m
% Demo of Monte Carlo integration from Wasserman p405
a = 0;
b = 1;
S = 10000;
xs = unifrnd(a,b,S,1);
samples = (b-a)*xs.^3;
Ihat = mean(samples)
se = sqrt(var(samples)/S)
