function y = gaussian_activation_function(x,m,xS)

P = diag(xS);

invP = inv(P);

y = exp(-.5 * (x - m)' * invP * (x - m));

return