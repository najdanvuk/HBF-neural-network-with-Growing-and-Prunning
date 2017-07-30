function N = randomize_data_V1(X,pct_of_data)

[ni,lenx] = size(X);                    % dimension of input vector
M = round(pct_of_data * lenx);

a = 1;

r = randperm(lenx) - 1  ;
N = a + r(1:M) ;

% N = unique(r);