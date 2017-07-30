function y = gaussian_activation_function_rbf(x,m,xS)

% P = diag(xS);

invxS = inv(xS);

y = exp(-.5 * (x - m)' * invxS * (x - m));

return