function [hbf,PARAMETERS ] = EKF_HBF_network_training_March_2013(X,Y,Xtest,Ytest,parameters)

epsilon_max = parameters.epsilon(1);
epsilon_min = parameters.epsilon(2);
eta = parameters.eta ;
learning_accuracy = parameters.learning_accuracy ;
p0 = parameters.p0;
q0 = parameters.q0;
r0 = parameters.r0;
xS0  = parameters.xS0;
N = parameters.N;
nn = parameters.nn;
n_pre_history  = parameters.n_pre_history;
pct_of_data = parameters.pct_of_data;
initial_number_of_neurons = parameters.initial_number_of_neurons;
switch_normalization = parameters.switch_normalization;
switch_random_perm = parameters.switch_random_perm;
gamma = parameters.gamma;
kappa = parameters.kappa;
Emin = eta * learning_accuracy;
q_norm = parameters.q_norm;
switch_matlab_GMM = parameters.switch_matlab_GMM;

%     normalize data into [0,1]
if switch_normalization
    XX = [X Xtest] ; YY = [Y Ytest] ;
    minx = min(XX(:)); maxx = max(XX(:));
    miny = min(YY(:)); maxy = max(YY(:));
    X = (X - minx)./(maxx-minx); Y = (Y - miny)./(maxy-miny);
    Xtest = (Xtest - minx)./(maxx-minx);Ytest = (Ytest - minx)./(maxx-minx);
end

% Start training
% start_time_train = cputime;
tic
% model input distribution
if switch_matlab_GMM == 1
    GMM_matlab = gmdistribution.fit(X(:,sort(n_pre_history))',3);
    bestpp = GMM_matlab.PComponents;
    bestmu = (GMM_matlab.mu)';
    bestcov = GMM_matlab.Sigma;
else
    [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures_NEW(X(:,sort(n_pre_history)),1,5,0,1e-4,0);
end

[rowX,colX] = size(X); [ rowY ,colY] = size(Y);
dim_x = 2 * rowX + rowY;

% %
P = p0 * eye(dim_x);                                 % state uncertainty
Q = q0 * eye(dim_x);                            % system uncertainty
R = r0 * eye(rowY);                           % measurement uncertainty

number_of_units = [];

%     randomize data and form training and test sets
if switch_random_perm
    X = X(:,N);Y = Y(:,N);
end
Xall = X; Yall = Y;     % all data
%     Xtest = X;Ytest = Y;    % test data
%     Xtest(:,N) = [];Ytest(:,N) = [];
%     X = Xtrain; Y = Ytrain;         % training data

% %

for kk = 1 : colX
    
    fprintf('Iteration #%d\n', kk);
    if kk == 1
        hbf.xW = Y(:,nn);
        hbf.xC = X(:,nn);
        xx = sqrt(X(:,kk)' * X(:,kk));
        hbf.xS = repmat((xS0 .* xx .* ones(rowX,1)),1,initial_number_of_neurons);
        hbf.cov = repmat(eye(dim_x) .* p0,1,initial_number_of_neurons);
    end
    
    epsilon(kk) = max(epsilon_max * (gamma ^ kk), epsilon_min);
    
    % %calculate the nearest neuron
    Euclid_distance = [];
    for i = 1 : size(hbf.xC,2)
        ed = sqrt((X(:,kk) - hbf.xC(:,i))'*inv(diag(hbf.xS(:,i)))*(X(:,kk) - hbf.xC(:,i)));
        Euclid_distance = [Euclid_distance ed];
    end
    
    [a,neuron] = find(Euclid_distance == min(Euclid_distance));
%     Euclid_distance(neuron)
%     epsilon(kk)
    %             if length(neuron) > 1, neuron = neuron(round(rand+1)); end% I have to
    %
    % % overall network output
    Act_fun = [];
    for jj = 1 :  size(hbf.xC,2)
        Act_fun(jj) = gaussian_activation_function(X(:,kk) , hbf.xC(:,jj) , hbf.xS(:,jj));
    end
    E(:,kk)= Y(:,kk) - hbf.xW * Act_fun';
    e(kk) = sqrt( E(:,kk)' * E(:,kk)) / rowY;
    
    % compute parameters of potential new neuron
    xWnew = E(:,kk);
    xCnew = X(:,kk);
    xSnew = (kappa * Euclid_distance(neuron) .* ones(rowX,1));
    
    % compute "significance" of new neuron
    Ntnew = [];
    for pp = 1 : length(bestpp)
        variablenew = xCnew - bestmu(:,pp);
        covariancenew = diag(xSnew) / q_norm + bestcov(:,:,pp);
        ntnew = gaussian_pd( variablenew, zeros(size(variablenew)) , covariancenew);
        Ntnew = [ Ntnew ntnew ] ;
    end
    NtAnew = Ntnew * bestpp';
    Esignew = norm(xWnew,q_norm) * ((2*pi/q_norm)^(rowX/2) * det(diag(xSnew ))^(.5) * NtAnew)^(1/q_norm);
    
    if ((Euclid_distance(neuron) > epsilon(kk)) && (Esignew > Emin))
%         kk
        disp 'add new neuron'
        hbf.xW = [hbf.xW xWnew];
        hbf.xC = [hbf.xC xCnew];
        hbf.xS = [hbf.xS xSnew];
        hbf.cov = [hbf.cov eye(dim_x) .* p0];
    else
%         kk
        disp 'update nearest neuron'
        % jacobian
        J = hbf_jacobian_jun_2012(...
            hbf.xW(:,neuron),X(:,kk),hbf.xC(:,neuron),...
            hbf.xS(:,neuron),rowY);
        % state vector
        x = [hbf.xW(:,neuron) ; hbf.xC(:,neuron) ; hbf.xS(:,neuron)];
        cov = hbf.cov(: , dim_x * neuron - dim_x + 1 : dim_x * neuron);
        cov = cov + Q;
        invS = inv(J' * cov * J + R);
        K = cov * J * invS;
        x = x + K * E(:,kk);
        cov = (eye(length(x)) - K * J') * cov;
        
        hbf.xW(:,neuron) = x(1:rowY);
        hbf.xC(:,neuron) =  x(rowY+1:rowY+rowX);
        hbf.xS(:,neuron) =  x(rowY+rowX+1:end);
        hbf.cov(:,dim_x * neuron - dim_x + 1 : dim_x * neuron) = cov;
        
        % check significance of adjusted / adapted neuron
        Nt = [];
        for pp = 1 : length(bestpp)
            variable = hbf.xC(:,neuron) - bestmu(:,pp);
            covariance = diag(hbf.xS(:,neuron)) / q_norm + bestcov(:,:,pp);
            nt = gaussian_pd( variable, zeros(size(variable)) , covariance);
            Nt = [ Nt nt ] ;
        end
        NtA = Nt * bestpp';
        Esig_nearest = norm(hbf.xW(:,neuron),q_norm) * ((2*pi/q_norm)^(rowX/2) * det(diag(hbf.xS(:,neuron)) )^(.5) * NtA)^(1/q_norm);
        if Esig_nearest < Emin
            if size(hbf.xC,2) == 1
                number_of_units = [number_of_units size(hbf.xC,2)];
                continue
            end
            hbf.xW(:,neuron) = [];
            hbf.xC(:,neuron) = [];
            hbf.xS(:,neuron) = [];
            hbf.cov(:,dim_x * neuron - dim_x + 1 : dim_x * neuron) = [];
            disp('pimpek! delete the neuron')
        else
            [Esig_nearest Emin];
            disp('neuron is significant!')
        end
    end
    %     end_time_train = toc;
    
    %     TrainingTime = end_time_train - start_time_train;
    number_of_units = [number_of_units size(hbf.xC,2)];
end
% disp('total number_of_units=')
% num2str(number_of_units(end))
TrainingTime = toc;
% output for the test set
thbf_test = [];
Yhbf_test = [];
for k = 1 : length(Xtest)
    for nn = 1 : size(hbf.xC,2)
        t_hbf = gaussian_activation_function(Xtest(:,k) ,hbf.xC(:,nn) , hbf.xS(:,nn));
        thbf_test = [thbf_test; t_hbf];
    end
    Yhbf_test = [Yhbf_test hbf.xW *thbf_test];
    thbf_test = [];
end

% output for the training data
thbf = [];
Yhbf = [];
for k = 1 : length(X)
    for nn = 1 : size(hbf.xC,2)
        t_hbf = gaussian_activation_function(X(:,k) ,hbf.xC(:,nn) , hbf.xS(:,nn));
        thbf = [thbf; t_hbf];
    end
    Yhbf = [Yhbf hbf.xW *thbf];
    thbf = [];
end

mse_test = mse(Yhbf_test - Ytest);
rmse_test = sqrt(mse_test)
mae_test = mae(Yhbf_test - Ytest);
sse_test = sse(Yhbf_test - Ytest);

mse_train = mse(Yhbf-Y);
rmse_train = sqrt(mse_train)
mae_train = mae(Yhbf-Y);
sse_train = sse(Yhbf-Y);

PARAMETERS.statistics = [mse_test rmse_test mae_test mse_train rmse_train mae_train];
PARAMETERS.e = e;
PARAMETERS.Yhbf = Yhbf;
PARAMETERS.Yhbf_test = Yhbf_test;
PARAMETERS.Ytest = Ytest;
PARAMETERS.number_of_units = number_of_units ;
PARAMETERS.TrainingTime = TrainingTime;
% Units = [Units number_of_units(end)];
% SSE_TRAIN = [SSE_TRAIN sse_train];
% SSE_TEST = [SSE_TEST sse_test];
%
% RMSE_TRAIN = [RMSE_TRAIN rmse_train];
% RMSE_TEST = [RMSE_TEST rmse_test];
% %
% MAE_TRAIN = [MAE_TRAIN mae_train];
% MAE_TEST = [MAE_TEST mae_test];
%
% MSE_TRAIN = [MSE_TRAIN mse_train];
% MSE_TEST = [MSE_TEST mse_test];