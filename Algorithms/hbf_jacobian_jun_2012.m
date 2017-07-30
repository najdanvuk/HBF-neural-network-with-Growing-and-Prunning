function Jacobian = hbf_jacobian_jun_2012(W,X,mu,xC,no)

% % W - weigths (vector of weights for nearest neuron)
% % X - input vector
% % mu - prototype vector
% % xC - vector of neuron widths
% % no - # of output dimensions

% jacobian wrt weights
Jac_weights = eye(no) * gaussian_activation_function(X, mu, xC);
% =========================================================================
% jacobian wrt centers
Jmu = [];
for m = 1 : no
    h = gaussian_activation_function(X, mu, xC) * inv(diag(xC)) * (X - mu);
    J = W(m,1) * h;
    Jmu = [Jmu J];
end
% =========================================================================
% % jacobian wrt widths
Jac_b = [];
for m = 1 : no
    bder =  .5 * W(m,1)  * (X - mu) .* (X - mu) * ...
            gaussian_activation_function(X, mu, xC) .* 1./ xC.^2;
    Jac_b = [Jac_b bder];
end
% % =========================================================================
% % and the winner is....
Jacobian = [Jac_weights; Jmu; Jac_b];