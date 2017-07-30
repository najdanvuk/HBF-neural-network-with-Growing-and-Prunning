% =========================================================================
% =========================================================================
% These files are free for non-commercial use. If commercial use is
% intended, you can contact me at e-mail given below. The files are intended
% for research use by experienced Matlab users and includes no warranties
% or services.  Bug reports are gratefully received.
% I wrote these files while I was working as scientific researcher
% with University of Belgrade - Innovation Center at Faculty of Mechanical
% Engineering so as to test my research ideas and show them to wider
% community of researchers. None of the functions are meant to be optimal,
% although I tried to (some extent) speed up execution. You are wellcome to
% use these files for your own research provided you mention their origin.
% If you would like to contact me, my e-mail is given below.
% =========================================================================
% If you use this code for your own research, please cite it as:
% Vukovic,N., Miljkovic,Z., A Growing and Pruning Sequential Learning
% Algorithm of Hyper Basis Function Neural Network for Function Approximation,
% Neural Networks, Vol. 46C, pp.210-226, Elsevier, October 2013.
% =========================================================================
% =========================================================================
% All rights reserved by: Dr. Najdan Vukovic
% contact e-mail: najdanvuk@gmail.com or nvukovic@mas.bg.ac.rs
% =========================================================================
% =========================================================================
clc,clear,close all
% This file simulates HBF- GP for Dynamical system with long input delays.
% =========================================================================
% Note: you need to add all folders to path (select folder, right click=> add folders...)
% =========================================================================
% =========================================================================
%               I N I T I A L I Z E   P A R A M E T E R S
% =========================================================================
% =========================================================================
% =========================================================================
% Test sets
% =========================================================================
% % Dynamical system with long input delays
dynamical_system_with_long_input_delays;
X = Xtrain'; Y = Ytrain';
Xtest = Xtest'; Ytest = Ytest';
[ rowX ,colX] = size(X);
% =========================================================================
%               I N I T I A L I Z E   P A R A M E T E R S
% =========================================================================
number_of_trials = 30
pre_history_data = .1 %pct of data used for GMM
pct_of_data = 1
% =========================================================================
% Thresholds for growing and prunning of HBF neurons. Please take a look at
% the paper for details.
epsilon_max_q2 = 1
epsilon_min_q2 = .01
epsilon_max_q1 = epsilon_max_q2
epsilon_min_q1 =  epsilon_min_q2
% =========================================================================
gamma = .999 % decay
eta = .1
learning_accuracy = 1
Emin = eta * learning_accuracy
initial_number_of_neurons = 1
q_norm_1 = 1
q_norm_2 = 2
% =========================================================================
% Which GMM are you using? Figueiredo's or Matlab's? I prefere Figueiredo's.
% Note: Figueiredo, M. A. T., & Jain, A. K. Unsupervised learning of finite mixture
% models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(3),
% 381–396.
switch_matlab_GMM = 0
% =========================================================================
switch_random_perm = 1
% =========================================================================
p0_ekf = 9e-1        % initial covariance for EKF state vector
q0_ekf = 1e-3        % state uncertainty
r0_ekf = 1e0        % measurement uncertainty
% =========================================================================
kappa = .4    %
xS0 =  kappa .* (max(X,[],2) - min(X,[],2))  %.* kappa           % initial spread of HBF width npr. norm(randn(size(Xtest,1),1))
% =========================================================================
% =========================================================================
% =========================================================================

%     HBF parameters
% %  q_norm = 1
parameters_ekf_1.epsilon(1) = epsilon_max_q1;
parameters_ekf_1.epsilon(2) = epsilon_min_q1;
parameters_ekf_1.eta = eta;
parameters_ekf_1.learning_accuracy = learning_accuracy;
parameters_ekf_1.p0 = p0_ekf;
parameters_ekf_1.q0 = q0_ekf;
parameters_ekf_1.r0 = r0_ekf;
parameters_ekf_1.xS0 = xS0;
parameters_ekf_1.pct_of_data = pct_of_data;
parameters_ekf_1.initial_number_of_neurons = initial_number_of_neurons;
parameters_ekf_1.switch_normalization = 0;                 % normalization
parameters_ekf_1.switch_random_perm = switch_random_perm ;
parameters_ekf_1.kappa = kappa;
parameters_ekf_1.gamma = gamma;
parameters_ekf_1.q_norm = q_norm_1 ;
parameters_ekf_1.switch_matlab_GMM = switch_matlab_GMM;
% %  q_norm = 2
parameters_ekf_2.epsilon(1) = epsilon_max_q2;
parameters_ekf_2.epsilon(2) = epsilon_min_q2;
parameters_ekf_2.eta = eta;
parameters_ekf_2.learning_accuracy = learning_accuracy;
parameters_ekf_2.p0 = p0_ekf;
parameters_ekf_2.q0 = q0_ekf;
parameters_ekf_2.r0 = r0_ekf;
parameters_ekf_2.xS0 = xS0;
parameters_ekf_2.pct_of_data = pct_of_data;
parameters_ekf_2.initial_number_of_neurons = initial_number_of_neurons;
parameters_ekf_2.switch_normalization = 0  ;               % normalization
parameters_ekf_2.switch_random_perm = switch_random_perm ;
parameters_ekf_2.kappa = kappa;
parameters_ekf_2.gamma = gamma;
parameters_ekf_2.q_norm = q_norm_2 ;
parameters_ekf_2.switch_matlab_GMM = switch_matlab_GMM;


% =========================================================================
% =========================================================================
%  ALLOCATE MATRICES FOR MEMORY AND TO SPEED UP THINGS A BIT
% =========================================================================

Units_ekf_1 = NaN(number_of_trials,1);
Stats_ekf_1 = NaN(number_of_trials,6); % [mse_test rmse_test mae_test mse_train rmse_train mae_train]
TrainingTime_1= NaN(number_of_trials,1);
Num_units_1 = NaN(number_of_trials,colX);

Units_ekf_2 = NaN(1,number_of_trials);
Stats_ekf_2 = NaN(number_of_trials,6); % [mse_test rmse_test mae_test mse_train rmse_train mae_train]
TrainingTime_2 = NaN(number_of_trials,1);
Num_units_2 = NaN(number_of_trials,colX)

RMS_1 = NaN(number_of_trials,colX) ;
RMS_2 = NaN(number_of_trials,colX) ;

YHBF_TEST_1 = NaN(number_of_trials,colX) ;
YHBF_TEST_2 = NaN(number_of_trials,colX) ;

% NOTE: you can add which ever statistics you want, just open
% EKF_HBF_network_training_March_2013 and define what you actually want...
% =========================================================================
% =========================================================================

for Step = 1 : number_of_trials
    % =====================================================================
    %
    fprintf('Step #%d\n', Step);
    
    N_ekf_1 = randomize_data_V1(X,pct_of_data);
    N_ekf_2 = randomize_data_V1(X,pct_of_data);
    
    nn_ekf_1 = randomize_data_V1(X , initial_number_of_neurons / colX);
    nn_ekf_2 = randomize_data_V1(X , initial_number_of_neurons / colX);
    
    n_pre_history = randomize_data_V1(X, pre_history_data );
    
    % ===============================================================
    %     HBF parameters
    % %  q_norm = 1
    
    parameters_ekf_1.N = N_ekf_1;
    parameters_ekf_1.nn = nn_ekf_1;
    parameters_ekf_1.n_pre_history = n_pre_history;
    % %  q_norm = 2
    parameters_ekf_2.N = N_ekf_2;
    parameters_ekf_2.nn = nn_ekf_2;
    parameters_ekf_2.n_pre_history = n_pre_history;
    
    % train HBF network
    %     EKF
    %     q_norm = 1
    [hbf_ekf_1,PARAMETERS_EKF_1_ ] = EKF_HBF_network_training_March_2013(X,Y,Xtest,Ytest,parameters_ekf_1);
    %     q_norm = 2
    [hbf_ekf_2,PARAMETERS_EKF_2_ ] = EKF_HBF_network_training_March_2013(X,Y,Xtest,Ytest,parameters_ekf_2);
    % % %
    disp('mse_test rmse_test mae_test mse_train rmse_train mae_train')
    PARAMETERS_EKF_1_.statistics
    PARAMETERS_EKF_2_.statistics
    
%     Just to see what actually is happens during learning
    figure(1) ; clf
    set(gca,'FontName','Arial','FontSize',12);
    subplot(3,1,1),plot(Ytest,'b'),hold on,
    plot(Ytest,'b')
    plot(PARAMETERS_EKF_1_.Yhbf_test,'r'),hold on
    plot(PARAMETERS_EKF_2_.Yhbf_test,'k'),hold on
    legend('Y','HBF - GP (1 norm)','HBF - GP (2 norm)');
    title('Test set'), hold on
    
    subplot(3,1,2),
    plot(PARAMETERS_EKF_1_.number_of_units,'r'),hold on
    plot(PARAMETERS_EKF_2_.number_of_units,'k');
    legend('HBF - GP (1 norm)','HBF - GP (2 norm)');
    title('# of units');
    
    subplot(3,1,3),
    plot(PARAMETERS_EKF_1_.e,'r'),hold on;
    plot(PARAMETERS_EKF_2_.e,'k'),hold on;
    legend('HBF - GP (1 norm)','HBF - GP (2 norm)');
    title('Error');
    pause(1)
    
    Units_ekf_1(Step,:) = PARAMETERS_EKF_1_.number_of_units(end);
    Stats_ekf_1(Step,:) = PARAMETERS_EKF_1_.statistics;
    TrainingTime_1(Step,:) = PARAMETERS_EKF_1_.TrainingTime;
    Num_units_1(Step,:) = PARAMETERS_EKF_1_.number_of_units;
    
    Units_ekf_2(Step,:) = PARAMETERS_EKF_2_.number_of_units(end);
    Stats_ekf_2(Step,:) = PARAMETERS_EKF_2_.statistics;
    TrainingTime_2(Step,:) = PARAMETERS_EKF_2_.TrainingTime;
    Num_units_2(Step,:) = PARAMETERS_EKF_2_.number_of_units;
    
    RMS_1(Step,:) = PARAMETERS_EKF_1_.e ;
    RMS_2(Step,:) = PARAMETERS_EKF_2_.e;
    
    YHBF_TEST_1(Step,:) = PARAMETERS_EKF_1_.Yhbf_test;
    YHBF_TEST_2(Step,:) = PARAMETERS_EKF_2_.Yhbf_test;
        
end

'mse_test rmse_test mae_test mse_train rmse_train mae_train'
mean_Stats_ekf_1 = mean(Stats_ekf_1)
std_Stats_ekf_1 = std(Stats_ekf_1)
mean_Units_ekf_1 = mean(Units_ekf_1)
std_Units_ekf_1 = std(Units_ekf_1)

mean_Stats_ekf_2 = mean(Stats_ekf_2)
std_Stats_ekf_2 = std(Stats_ekf_2)
mean_Units_ekf_2 = mean(Units_ekf_2)
std_Units_ekf_2 = std(Units_ekf_2)

figure(33),clf
set(gca,'FontName','Arial','FontSize',12);
plot((round(mean(Num_units_1))),'-r','LineWidth', 4),hold on
plot((round(mean(Num_units_2))),'-k','LineWidth', 2),hold on
xlabel('Number of Observations');
ylabel('Number of Neurons'); axis([0 500 0 30])
legend('HBF - GP (1 norm)','HBF - GP (2 norm)');
title('# of units');

'RMSE_TEST'
'HBF_1_norm  HBF_2_norm   '
[mean_Stats_ekf_1(2) mean_Stats_ekf_2(2) ]
'Units'
[mean_Units_ekf_1  mean_Units_ekf_2   ]


figure(3),clf
set(gca,'FontName','Arial','FontSize',12);
plot(mean(Num_units_1),'LineWidth', 2),hold on
plot(mean(Num_units_2),'LineWidth', 2),hold on
xlabel('Number of Observations');
ylabel('Number of Neurons');
legend('1-norm','2-norm')

% Plot average perofrmance of HBF GP network
figure(10);clf
set(gca,'FontName','Arial','FontSize',12);
plot(Ytest-mean(YHBF_TEST_1),'k'),hold on
plot(Ytest-mean(YHBF_TEST_2),'r')
legend('HBF - GP (1 norm)','HBF - GP (2 norm)');
%
figure(13);clf
set(gca,'FontName','Arial','FontSize',12);
plot(Ytest,'--b'),hold on,
plot(mean(YHBF_TEST_1),'r','linewidth',2);
xlabel('Time step'),ylabel('Output')
legend('Y','HBF - GP (1 norm)');title('Test set');

figure(14);clf
set(gca,'FontName','Arial','FontSize',12);
plot(Ytest,'--b'),hold on,
plot(mean(YHBF_TEST_2),'k','linewidth',2);
xlabel('Time step'),ylabel('Output')
legend('Y','HBF - GP (2 norm)');title('Test set');

figure(155); clf
set(gca,'FontName','Arial','FontSize',12);
plot(mean(RMS_1),'r'),hold on;
plot(mean(RMS_2),'k'),hold on;
xlabel('Number of Observations'); ylabel('RMS Error');
legend('HBF - GP (1 norm)','HBF - GP (2 norm)')

% =========================================================================
% =========================================================================
% Note: ocasionally, numerical instabilities cause program to malfunction.
% Just stop it (Ctrl+c) and run it again. That is known bug yet to be
% fixed.
% =========================================================================
% =========================================================================