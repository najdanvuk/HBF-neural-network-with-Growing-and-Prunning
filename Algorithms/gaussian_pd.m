function prob_gauss = gaussian_pd(x, mu, C)

% dimension of vector x
d = length(mu);
% C = (C+C')*.5;
% define quadratic form
xg = x - mu; 
invC = inv(C);

% define coefficients
eta1 = (2*pi)^(d/2); eta2 = det(C)^(1/2);
eta = 1 / (eta1*eta2);

% calculate "gaussian"
prob = exp(-.5 * xg' * invC * xg);

% return prob_gauss
prob_gauss = eta * prob;
return