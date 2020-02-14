function [ MCMC, Regime_dist, beta_samps] = MRS_MCMC_FN(data,N_samps,n_adapt,model,startR,percent_R_changes, check_plots, Priors, AR1Design,Beta_Priors,isWaveletModel)
%MRS_MCMC_FN implements an adaptive blockwise data augmented MCMC to
%   sample from the posterior of a Markovian regime-switching model with
%   independent regimes. It creates 4 MCMC chains in parallel. This
%   function is only used to call MRS_MCMC_HPC_par, which does all the
%   work.
%
%   Outputs:
%   MCMC - an (N_samps x #parameters X 4) array containing MCMC chains.
%   Each column corresponds to a parameter of our model and the third
%   dimension corresponds a different MCMC chain.
%   Regime_dist - A (length(data) X n_regimes) array containing posterior
%   probabilities. The (i,j) element of Regime_dist contains the
%   probability P(data(i) is from regime j |data).
%
%   Inputs:
%   data - the data you want to model as a column vector.
%   N_samps - a scalar specifying how long you want each MCMC chain to be
%   n_adapt - the number of adaptive/burn-in iterations you want
%   model - a cell array containing strings specifying the model. NOTE all
%   priors are uniform (and may be improper)
%       eg 1. for an MRS model with one AR(1) base regime, one spike regime
%       and one drop regime we would specify
%       model = {{'AR(1)'}{'LN shifted' 'LN shifted and reversed'}}
%       eg 2. for an MRS model with two AR(1) base regimes, two spike
%       regimes and one drop regime we would specify
%       model = {{'AR(1)' 'AR(1)'}{'LN shifted' 'LN sifted' 'LN shifted and reversed'}}
%       Notes:
%       - Only put 'AR(1)'s in the first element of model
%       - In the second element of model we can choose to specify any of
%           'G' ~ Normal distribution
%           'LN' ~ log-normal distribution
%           'LN shifted' ~ shifted lognormal distribution
%           'LN shifted and reversed' ~ shifted and reversed lognormal
%           distribution
%           'Gamma' ~ Shifted gamma distribution. Same shifting mechanism
%           as the shifted lognormal distribution.
%       - Where more than 1 'AR(1)' regime is specified the we resrict the
%       variances of each model for identifiability. \sigma_1<\sigma_2<...
%       - We may only specify 2 'LN shifted' regimes. The prior for the
%       first 'LN shifted' regime has support from
%       [66.666%ile of data, \infty) and the prior for the second
%       'LN shifted' regime (when specified) is from
%       [98%il of data, \infty).
%       - We may only specify 1 'LN shifted and reversed' regime. The
%       support for this prior is (-\infty, 33.333%ile of data]
%       - You can change these supports by specifying new values for q1 q2
%       and q3
%   startR - a starting point for the regime sequence in our MCMC. A vector
%   of length(data). May be specified as empty, [].
%   percent_R_changes - a scalar in (0,1], the % of changes we want to try
%   and make to the hidden regime sequence for each iteration in our MCMC.
%   I find that 0.1 works well for the SA dataset and this is the default.
%   If percent_R_changes is close to 1 mixing is fast but the algorithm is
%   slow, if percent_R_changes is close to 0 mixing is slow by the
%   algorithm is faster.
%   check_plots - true OR false. specify true if you want to see checking
%   plots.
%
%   Example. 
%
%       N_MCMC_samples = 1500000; % specify the number of MCMC samples to use
%                                 % in total.
%
%       N_adapt_samples = 500000; % the number of adative samples to make.
%
%       model = {{'AR(1)'} {'Gamma' 'Gamma'}}; % A model with 3 reigmes, one
%                                              % AR(1) base regime, and two 
%                                              % spike regimes, one with a 
%                                              % higher shifitng parameter than
%                                              % the other.
%
%       start_R = randi([0,1],size(data)); % specify a random starting
%                                          % regime sequence.
%                                                                         
%       percent_R_proposed = 0.1; % propose changes to 10% of regimes at
%                                 % each sweep of the algorithm.
%
%       plots = false; % Don't make PPC plots
%
%       [mcmc,R_dist] = MRS_MCMC_FN(data,N_MCMC_samples,N_adapt_samples,model,start_R,plots);%
%
%
% Written by Angus Lewis Nov/2017


global n_regimes
n_chains = 4; % you can change this if you have more cores at your disposal
AR_model = model{1};
IR_model = model{2};
n_AR = length(AR_model);
n_IR = length(IR_model);
n_regimes = n_AR+n_IR;
r = length(data);
n_params = 3*n_AR+2*n_IR;
n_q = 0;
for i = 1:n_IR
    if strcmp(IR_model{i},'LN shifted')
        n_q = n_q+1;
    elseif strcmp(IR_model{i},'Gamma')
        n_q = n_q+1;
    elseif strcmp(IR_model{i},'LN shifted and reversed')
        n_q = n_q+1;
    end
end
idx = false(1,length(Priors)); 
idx(n_AR*3 + (1:n_IR)*3-2) = true;
qPriors = Priors(idx); 
idx = ~idx;
ParamPriors = Priors(idx);

MCMC = nan([N_samps,n_params+(n_AR+n_IR)^2+n_q,n_chains]);
if check_plots == true
    plots_R_store = nan([r,5,n_chains]);
    plots_index_store = nan(5,n_chains);
end
beta_samps = nan(N_samps,size(AR1Design,2),n_chains);
R_dist_storage = nan([r,n_regimes,n_chains]);

% This loop creates 4 parallel chains of our MCMC, parfor
for i = 1:n_chains
    [ MCMC_theta, MCMC_P , R_dist, plots_index, plots_R,beta_params] = MRS_MCMC_HPC_par(data,N_samps,model,n_adapt,startR,percent_R_changes,qPriors,ParamPriors, AR1Design,Beta_Priors,isWaveletModel);
    MCMC(:,:,i) = [MCMC_theta,MCMC_P];
    beta_samps(:,:,i) = beta_params; 
    if check_plots == true
        plots_R_store(:,:,i) = plots_R;
        plots_index_store(:,i) = plots_index;
    end
    R_dist_storage(:,:,i) = R_dist;
end
Regime_dist = sum(R_dist_storage,3)/n_chains;

% checking
if check_plots == true
    checking_plots(data,MCMC(plots_index_store(:,1),:,:),plots_R_store,n_AR,n_IR,model,AR1Design,beta_samps(plots_index_store(:,1),:,:));
end
end

function[ MCMC_theta, MCMC_P , R_dist, plots_index, plots_R,beta_params] = MRS_MCMC_HPC_par(data,N_samps,model,n_adapt,startR,percent_R_changes,qPriors,ParamPriors, AR1Design,Beta_Priors, isWaveletModel)
% MRS_MCMC_HPC_par implements our MCMC. It does all the work.
% if the number of adaptive iterations is not specifiedor is 0, then the
% whole chain is adaptive (and therefore not Markovian), but shold still
% converge to the posterior.

global n_regimes

if nargin < 4
    n_adapt = Inf;
end

% Extract model specifications
AR_model = model{1};
IR_model = model{2};
n_AR = length(AR_model);
n_IR = length(IR_model);
n_regimes = n_AR+n_IR;
q1 = quantile(data,1/3); % change these if you want to change the support
q2 = quantile(data,2/3); % of the shifting parameters
q3 = quantile(data,0.98);
% initialise some space in memory
r = length(data);
n_params = 3*n_AR+2*n_IR;
chain = [nan(N_samps,n_params),nan(N_samps,(n_AR+n_IR)^2)];
beta_params = zeros(N_samps,size(AR1Design,2));
cumulative_R = zeros(r,n_AR+n_IR);
c1 = 1;
new_R_inds = zeros(r,n_AR+n_IR);
thin_stat = 100;
n_plots = 5;
thin_plot = max(floor((N_samps - n_adapt)/thin_stat/n_plots),1);
n_stats = floor((N_samps-n_adapt)/thin_stat);
stats_index = nan(1,n_stats-1);
stats_R = nan(r,n_stats-1);
plots_index = nan(1,n_plots-1);
plots_R = nan(r,n_plots-1);
c2 = 1;
c3 = 1;

n_q = 0;
q_init = zeros(1,n_IR);
regime_spec = [];
for i = 1:n_IR
    if strcmp(IR_model{i},'LN shifted')
        n_q = n_q+1;
        if n_q == 1
            q_init(i) = q2;
        else
            q_init(i) = q3;
        end
        regime_spec = [regime_spec,n_AR + i - 1];
    elseif strcmp(IR_model{i},'Gamma')
        n_q = n_q+1;
        if n_q == 1
            q_init(i) = q2;
        else
            q_init(i) = q3;
        end
        regime_spec = [regime_spec,n_AR + i - 1];
    elseif strcmp(IR_model{i},'LN shifted and reversed')
        n_q = n_q+1;
        q_init(i) = q1;
        regime_spec = [regime_spec,n_AR + i - 1];
    end
end
q_params = [q_init; zeros(N_samps-1,length(q_init))];

% initialise starting points for parameters
for i = 0:(n_AR-1)
    ind = 3*i + 1; index = ind:(ind+2);
    chain(1,index) = [0,0.5,1];
end
for i = n_AR:(n_regimes-1)
    ind = 3*n_AR + (i-n_AR)*2 + 1; index = ind:(ind+1);
    chain(1,index) = [2,2];
end
if isWaveletModel{1}
    beta_params(1,:) = ([isWaveletModel{2}]'*[isWaveletModel{2}])^-1*[isWaveletModel{2}]'*isWaveletModel{6}(:);
else
    beta_params(1,:) = ([AR1Design]'*[AR1Design])^-1*[AR1Design]'*data(:);
end

if nargin <5 || isempty(startR)
    R = zeros(size(data)); % Default starting point for chain
else
    R = startR;
end
R(1) = 0;
R(end) = 0;

% Specify number of changes to make to the hidden sequence
if nargin < 6
    percent_R_changes = 0.1;
end

% initial variances for proposals
sigma = [ones(1,n_params),ceil(0.1*r)];
sigmaq = ones(1,n_q);
beta_sigma = ones(1,size(AR1Design,2));

% Now we can start the MCMC chain
for n = 2:N_samps
    if mod(n,ceil(N_samps/10)) == 0
        n/N_samps
        %plot(AR1Design*beta_params(n-1,:)'), hold on, drawnow
    end
    % MCMC sampling of p_ij parameters
    P = p_ij_sampler(R, n_AR+n_IR);
    chain(n,n_params+1:end) = P(:);
    
    % MCMC sample of the regime sequence
    R = R_sampler(R,sigma(end),P,data,chain(n-1,:),n_AR,n_IR, IR_model, q_params(n-1,:),AR1Design,beta_params(n-1,:));
    if n >= n_adapt
        for l = 0:(n_IR+n_AR-1)
            new_R_inds(:,l+1) = (R==l);
        end
        cumulative_R = new_R_inds + cumulative_R;
    end
    
    % MCMC sample cut-offs
    if n_q > 0
        q_params(n,:) = q_sampler(q_params(n-1,:),data,R,chain(n-1,:),IR_model,n_q,regime_spec,n_AR,sigmaq,qPriors);
    end
    
    % MCMC sampling of the parameters
    for i = 0:(n_AR-1) % AR(1) parameters
        ind = 3*i + 1; index = ind:(ind+2);
        chain(n,index) = AR_sampler(chain(n-1,index),data,R,sigma(index),i,n_AR,chain(n-1,:),i+1,ParamPriors(index), AR1Design, beta_params(n-1,:));
    end
    if n_AR>0
        beta_params(n,:) = beta_sampler(chain(n-1,:),data,R,beta_sigma,n_AR,Beta_Priors, AR1Design, beta_params(n-1,:),isWaveletModel);
    end
    for i = n_AR:(n_AR+n_IR-1) % Independent regime parameters
        ind = 3*n_AR + (i-n_AR)*2 + 1; index = ind:(ind+1);
        chain(n,index) = IR_sampler(chain(n-1,index),data,R,sigma(index),i, IR_model{i-n_AR+1},q_params(n,:),ParamPriors(index),n_AR);
    end
    
    % Automatic tuning
    if mod(n,50) == 0 && n <= n_adapt
        for k = 1:(n_params+(n_AR+n_IR)^2)
            test_chain = chain(n-49:n,k);
            % Adjust variances for params
            if k <= n_params
                if n > 50
                    acc_rate = length(unique(test_chain))/50;
                else
                    acc_rate = length(unique(test_chain))/n;
                end
                if acc_rate < 0.44
                    sigma(k) = max(sigma(k)*exp(-min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
                elseif acc_rate > 0.44
                    sigma(k) = max(sigma(k)*exp(min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
                end
            end
        end
        for k = 1:n_q
            test_chain = q_params(n-49:n,k);
            % Adjust variance for shifting params
                if n > 50
                    acc_rate = length(unique(test_chain))/50;
                else
                    acc_rate = length(unique(test_chain))/n;
                end
                if acc_rate < 0.44
                    sigmaq(k) = max(sigmaq(k)*exp(-min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
                elseif acc_rate > 0.44
                    sigmaq(k) = max(sigmaq(k)*exp(min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
                end
        end
        for k = 1:size(AR1Design,2)
            test_chain = beta_params(n-49:n,k);
            if n > 50
                acc_rate = length(unique(test_chain))/50;
            else
                acc_rate = length(unique(test_chain))/n;
            end
            if acc_rate < 0.44
                beta_sigma(1,k) = max(beta_sigma(1,k)*exp(-min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
            elseif acc_rate > 0.44
                beta_sigma(1,k) = max(beta_sigma(1,k)*exp(min(1*[c1^(-1/2),10/(c1),10000/(c1^2)])),0.02);
            end
        end
        c1 = c1 + 1;
    end
    
    % Save info for creating PPC's
    if n>n_adapt
        if mod(n,thin_stat) == 0
            stats_index(c2) = n;
            stats_R(:,c2) = R;
            if mod(c2,thin_plot) == 0
                plots_index(c3) = n;
                plots_R(:,c3) = R;
                c3 = c3 + 1;
            end
            c2 = c2+1;
        end
    end
end

% Rearrange variables ready for output
R_dist = cumulative_R./(N_samps-n_adapt+1);
MCMC_theta = nan(N_samps,n_params+n_q);
MCMC_theta(:,1:n_AR*3) = chain(:,1:n_AR*3);
c4 = n_AR*3;
c5 = 1;
c6 = n_AR*3;
for i = 1:n_IR
    if strcmp(IR_model{i},'LN shifted') || strcmp(IR_model{i},'LN shifted and reversed') || strcmp(IR_model{i},'Gamma')
        MCMC_theta(:,c4+1:c4+3) = [q_params(:,c5),chain(:,c6+1:c6+2)];
        c5 = c5+1;
        c4 = c4+3;
        c6 = c6+2;
    else
        MCMC_theta(:,c4+1:c4+2) = [chain(:,c6+1:c6+2)];
        c4 = c4+2;
        c6 = c6+2;
    end
end
MCMC_P = chain(:,n_params+1:end);
end


% Internally used routines
function [ P ] = p_ij_sampler( R, n_regimes )
%P_IJ_SAMPLER uses Gibbs sampling to sample the transition probabilities
% from the posterior of a Markov chain.
%   R is the current Markov chain realisation
%   P is a Gibbs sample of the posterior of the Markov chain
%   the prior is flat on [0,1] for the p_ij's

l = n_regimes;
n = length(R)-1;
p = zeros(l,l);
for t = 1:n
    p(R(t)+1, R(t + 1)+1) = p(R(t)+1, R(t + 1)+1) + 1;
end

P = zeros(l,l);
for k = 1:l
    P(k,:) = drchrnd(p(k,:)+1,1);
end

end

function r = drchrnd(a,n)
%DRCHRND samples from a dirichlet distribution
%   a is a vector of parameters
%   n is the desired number of samples
%   r are random numbers from dirichlet(a)

a = a(:)';
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);

end

function [ R_new, acc ] = R_sampler( R_old, n, P, data, theta, n_AR, n_IR, IR_model, qs, AR1Design, beta_params )
% R_sampler takes and old sequence of regimes, R_old, and samples a new
% sequence, R_new, using MH. n specifies the number of elements of R_old
% that this function will try and change. P is the transition matrix of the
% Markov chain that governs the evolution of the regimes. P is used to
% specify a Markov chain prior for the regimes. Theta is a vector of
% parameters for the model.
l = n_AR+n_IR;
u = 0:(l-1);
unifdist = 1/(l-1):1/(l-1):1;
canR = R_old;
r = length(R_old);
index = randperm(r-2,n)+1;
acc = 0;
for ind = index
    % propose a change to any other regime
    R_t = R_old(ind);
    R_t_compliment = u(u~=R_t);
    sample_index = find(rand<unifdist,1);
    canR(ind) = R_t_compliment(sample_index);
    
    before = R_old(ind-1);
    after = R_old(ind+1);
    
    difflikes = diff_loglike_1Rchange(data,R_old,theta,n_AR,n_IR, IR_model, ind, canR(ind), qs, AR1Design, beta_params);
    if log(rand) < difflikes ...
            +log(P(before+1,canR(ind)+1)*P(canR(ind)+1,after+1)) ...
            -log(P(before+1,R_t+1)*P(R_t+1,after+1))
        R_old(ind) = canR(ind);
        acc = acc+1;
    else
        canR(ind) = R_old(ind);
    end
end

R_new = R_old;
end

function [ diff_ll ] = diff_loglike_1Rchange( data,R,theta, n_AR, n_IR, IR_model,t, change_to, qs, AR1Design, beta_params )
% Finds the log of the ratio of the likelihoods which differ only by 1
% point in the regimes.

r = length(R);

if any(R(t) == 0:n_AR-1)
    % if the current regime is an AR process, find the last time in that regime
    for j = (t-1):-1:1
        if R(j) == R(t)
            break
        end
    end
    % if the current regime is an AR process, find the next time in that regime
    for k = (t+1):r
        if R(k) == R(t)
            break
        end
    end
    params = theta(R(t)*3+1:R(t)*3+3);
    l1 = ARpdf(data(t),data(j),t-j,params, AR1Design(t,:), AR1Design(j,:), beta_params);
    l2 = ARpdf(data(k),data(t),k-t,params, AR1Design(k,:), AR1Design(t,:), beta_params);
    l3 = ARpdf(data(k),data(j),k-j,params, AR1Design(k,:), AR1Design(j,:), beta_params);
    ll = l1 + l2 - l3;
else
    params = theta(n_AR*3+(R(t)-n_AR)*2+1:n_AR*3+(R(t)-n_AR+1)*2);
    switch IR_model{R(t)-n_AR+1}
        case 'G'
            ll = - 0.5*log(2*pi*params(2)) - ...
                (data(t)-params(1)).^2./(2*params(2));
        case 'LN'
            if data(t) > 0
                ll = -log( data(t)*sqrt(2*pi*params(2)) )-...
                    ( log(data(t))-params(1) ).^2 ./( 2*params(2) );
            else
                ll = -Inf;
            end
        case 'LN shifted and reversed'
            q1 = qs(R(t)+1-n_AR);
            if q1>data(t)
                ll = -log( (q1-data(t))*sqrt(2*pi*params(2)) )-...
                    ( log( q1-data(t) )-params(1) ).^2 ./( 2*params(2) );
            else
                ll = -Inf;
            end
        case 'LN shifted'
            q2 = qs(R(t)+1-n_AR);
            if q2 < data(t)
                ll = -log( (-q2+data(t))*sqrt(2*pi*params(2)) )-...
                    ( log(-q2+data(t))-params(1) ).^2 ./( 2*params(2) );
            else
                ll = -Inf;
            end
        case 'Gamma'
            q2 = qs(R(t)+1-n_AR);
            if data(t) > q2
                ll = -log(params(2)^params(1)) ...
                    -gammaln(params(1))...
                    +(params(1)-1)*log(data(t)-q2)...
                    -(data(t)-q2)/params(2);
            else
                ll = -Inf;
            end
    end
end

if any(change_to == 0:n_AR-1)
    % if the current regime is an AR process, find the last time in that regime
    for J = (t-1):-1:1
        if R(J) == change_to
            break
        end
    end
    % if the current regime is an AR process, find the next time in that regime
    for K = (t+1):r
        if R(K) == change_to
            break
        end
    end
    PARAMS = theta(change_to*3+1:change_to*3+3);
    L1 = ARpdf(data(t),data(J),t-J,PARAMS, AR1Design(t,:), AR1Design(J,:), beta_params);
    L2 = ARpdf(data(K),data(t),K-t,PARAMS, AR1Design(K,:), AR1Design(t,:), beta_params);
    L3 = ARpdf(data(K),data(J),K-J,PARAMS, AR1Design(K,:), AR1Design(J,:), beta_params);
    LL = L1 + L2 - L3;
else
    PARAMS = theta(n_AR*3+(change_to-n_AR)*2+1:n_AR*3+(change_to-n_AR+1)*2);
    switch IR_model{change_to-n_AR+1}
        case 'G'
            LL = - 0.5*log(2*pi*PARAMS(2)) - ...
                (data(t)-PARAMS(1)).^2./(2*PARAMS(2));
        case 'LN'
            if data(t) > 0
                LL = -log( data(t)*sqrt(2*pi*PARAMS(2)) )-...
                    ( log(data(t))-PARAMS(1) ).^2 ./( 2*PARAMS(2) );
            else
                LL = -Inf;
            end
        case 'LN shifted and reversed'
            q1 = qs(change_to+1-n_AR);
            if q1 > data(t)
                LL = -log( (q1-data(t))*sqrt(2*pi*PARAMS(2)) )-...
                    ( log(q1-data(t))-PARAMS(1) ).^2 ./( 2*PARAMS(2) );
            else
                LL = -Inf;
            end
        case 'LN shifted'
            q2 = qs(change_to+1-n_AR);
            if q2 < data(t)
                LL = -log( (-q2+data(t))*sqrt(2*pi*PARAMS(2)) )-...
                    ( log(-q2+data(t))-PARAMS(1) ).^2 ./( 2*PARAMS(2) );
            else
                LL = -Inf;
            end
        case 'Gamma'
            q2 = qs(change_to+1-n_AR);
            if data(t) > q2
                LL = -log(PARAMS(2)^PARAMS(1)) ...
                    -gammaln(PARAMS(1))...
                    +(PARAMS(1)-1)*log(data(t)-q2)...
                    -(data(t)-q2)/PARAMS(2);
            else
                LL = -Inf;
            end
    end
end

diff_ll = -ll+LL;

end

function q_samps = q_sampler(q_params,data,R,theta,IR_model,n_q,regime_spec,n_AR,sigmaq,qPriors)
% This function samples the shifting parametrs
for i = 1:n_q
    r1 = rand;
    r2 = rand;
    q_can = sqrt(-2*log(r1))*cos(2*pi*r2)*sigmaq(i) + q_params(regime_spec(i)+1-n_AR);
    ind = 3*n_AR + (regime_spec(i)-n_AR)*2 + 1; index = ind:(ind+1);
    params = theta(index);
    prior = qPriors{i};
    difflikes = diff_loglike_qs(data,R,params,params,regime_spec(i),IR_model{regime_spec(i)-n_AR+1},q_params(regime_spec(i)+1-n_AR),q_can);
    if log(rand) < difflikes + log(prior(q_can)) - log(prior(q_params(regime_spec(i)+1-n_AR)))
        q_samps(i) = q_can;
    else
        q_samps(i) = q_params(regime_spec(i)+1-n_AR);
    end
end
end

function [ difflikes ] = diff_loglike_qs( data,R,theta,can,regime,IR_model,q_params,q_can )
% calcualtes canloglike - loglike when only the qs of LN shifted spike
% regime change
% data are the observed prices.
% theta are the old parameters.
% can are the new parameters = old parameters.
% regime specified which regime corresponds to an AR regime.
if any(q_can < min(data)) || any(q_can>max(data))
    difflikes = -Inf;
else
    data1 = data(R==regime); % the spikes
    llcan = -Inf;
    ll = -Inf;
    if ~isempty(data1)
        switch IR_model
            case 'LN shifted and reversed'
                if q_can > 0
                    ps = 0;
                    pscan = -Inf;
                else
                    old_dat = q_params-data1;
                    can_dat = q_can-data1;
                    
                    if any(old_dat(:) <= 0 )
                        ps = Inf;
                        pscan = 0;
                    elseif any(can_dat(:) <= 0 )
                        pscan = -Inf;
                        ps = 0;
                    else
                        ps = -log( (old_dat) *sqrt(2*pi*theta(2)) )-...
                            ( log((old_dat))-theta(1) ).^2 ./( 2*theta(2) );
                        pscan = -log( (can_dat)*sqrt(2*pi*can(2)) )-...
                            ( log((can_dat))-can(1) ).^2 ./( 2*can(2) );
                    end
                end
            case 'LN shifted'
                if q_can < 0
                    ps = 0;
                    pscan = -Inf;
                else
                    old_dat = data1-q_params;
                    can_dat = data1-q_can;
                    if any(old_dat(:) <= 0 )
                        ps = Inf;
                        pscan = 0;
                    elseif any(can_dat(:) <= 0 )
                        pscan = -Inf;
                        ps = 0;
                    else
                        ps = -log( (old_dat)*sqrt(2*pi*theta(2)) )-...
                            ( log((old_dat))-theta(1) ).^2 ./( 2*theta(2) );
                        pscan = -log( (can_dat)*sqrt(2*pi*can(2)) )-...
                            ( log((can_dat))-can(1) ).^2 ./( 2*can(2) );
                    end
                end
           case 'Gamma'
                if q_can < 0
                    ps = 0;
                    pscan = -Inf;
                else
                    old_dat = data1-q_params;
                    can_dat = data1-q_can;
                    if any(old_dat(:) <= 0 )
                        ps = Inf;
                        pscan = 0;
                    elseif any(can_dat(:) <= 0 )
                        pscan = -Inf;
                        ps = 0;
                    else
                        ps = -log(theta(2)^theta(1)) ...
                            -gammaln(theta(1))...
                            +(theta(1)-1)*log(old_dat)...
                            -(old_dat)/theta(2);
                        pscan = -log(can(2)^can(1)) ...
                            -gammaln(can(1))...
                            +(can(1)-1)*log(can_dat)...
                            -(can_dat)/can(2);
                    end
                end
        end
        llcan = sum(pscan);
        ll = sum(ps);
    end
    difflikes = llcan - ll;
end
end

function like = ARpdf(data, lagged_data, lag_size, params, AR1Design, AR1Designlagged, beta_params)
% Calculates the likelihood of f(data(t)|lagged_data) for a lag size of
% lag_size.
B = params(2).^(lag_size);
C = (1-B)./(1-params(2)); C = C(:);
mu = params(1)*C + AR1Design*beta_params(:) + params(2).^lag_size.*(lagged_data-AR1Designlagged*beta_params(:));
B2 = params(2).^(2*lag_size);
C2 = (1-B2)./(1-params(2)^2); C2 = C2(:);
s = C2*params(3);
like = -0.5*log(2*s*pi) - ((data-mu).^2)./(2*s);
end

function [ theta_new, acc ] = AR_sampler( theta_old, data, R, sigma, regime, n_AR, all_params, which_AR_regime,ParamPriors, AR1Design, beta_params )
%AR_SAMPLER is a function that returns a single sample from a MH step of
%the posterior for the parameters of an AR(1) regime in an MRS model
%   theta_old is a vector of length 3 containing the current samples of the
%   MCMC chain.
%   data is a vector of observed prices.
%   R is the regime sequence.
%   sigma is a vector of length 3 that specifies the proposal variances
%   theta_new is the new position of the MCMC chain also length 3.
% the parameters in the theta's (theta_new and theta_old) are specified as
% x_t = theta(1) + theta(2)x_{t-1} + theta(3)e_t
% All priors are improper and flat.


can = theta_old;
acc = zeros(size(theta_old));

% % propose new points from a normal distribution
r1 = rand;
r2 = rand;
% can(1) = sqrt(-2*log(r1))*cos(2*pi*r2)*sigma(1) + theta_old(1);
% 
% % accept/reject
% difflikes = diff_loglike_AR( data,R,theta_old,can,regime, AR1Design, beta_params );
% alpha = difflikes;
% prior = ParamPriors{1};
% 
% if log(rand) < alpha + log(prior(can(1))) - log(prior(theta_old(1)))
%     theta_old(1) = can(1);
%     acc(1) = 1;
% else
%     can(1) = theta_old(1);
% end

% propose new points from a normal distribution
can(2) = sqrt(-2*log(r1))*sin(2*pi*r2)*sigma(2) + theta_old(2);

% accept/reject
difflikes = diff_loglike_AR( data,R,theta_old,can,regime, AR1Design, beta_params );
alpha = difflikes;
prior = ParamPriors{2};

if log(rand) < alpha + log(prior(can(2))) - log(prior(theta_old(2)))
    theta_old(2) = can(2);
    acc(2) = 1;
else
    can(2) = theta_old(2);
end

if n_AR < 2
    % propose new points from a normal distribution
    r1 = rand;
    r2 = rand;
    can(3) = sqrt(-2*log(r1))*cos(2*pi*r2)*sigma(3) + theta_old(3);
    prior = ParamPriors{3};
    % accept/reject
    difflikes = diff_loglike_AR( data,R,theta_old,can,regime, AR1Design, beta_params );
    alpha = difflikes;
    if log(rand) < alpha + log(prior(can(3))) - log(prior(theta_old(3)))
        theta_old(3) = can(3);
        acc(3) = 1;
    end
else
    % propose new points from a normal distribution
    sigma_params = all_params(3:3:3*n_AR);
    r1 = rand;
    r2 = rand;
    can(3) = sqrt(-2*log(r1))*cos(2*pi*r2)*sigma(3) + theta_old(3);
    % accept/reject
    lower_params = sigma_params(1:which_AR_regime-1);
    upper_params = sigma_params(which_AR_regime+1:end);
    if isempty(lower_params)
        lower_params = -Inf;
    end
    if isempty(upper_params)
        upper_params = Inf;
    end
    if all(can(3) >= lower_params) && all(can(3) <= upper_params)
        difflikes = diff_loglike_AR( data,R,theta_old,can,regime, AR1Design, beta_params );
        alpha = difflikes;
        prior = ParamPriors{3};
        if log(rand) < alpha + log(prior(can(3))) - log(prior(theta_old(3)))
            theta_old(3) = can(3);
            acc(3) = 1;
        end
    end
end

theta_new = theta_old;
end

function [ difflikes ] = diff_loglike_AR( data,R,theta,can,regime, AR1Design, beta_params )
% calcualtes canloglike - loglike for a change only in the AR paremeters.
% data are the observed prices.
% theta are the old parameters.
% can are the new parameters.
% the parameters in the theta and can are specified as
% x_t = theta(1) + theta(2)x_{t-1} + theta(3)e_t
% regime specified which regime corresponds to an AR regime
if any([ can(3) <= 0 , abs(can(2)) > 1])
    difflikes = -Inf;
else
    ind = R==regime;
    lag0 = diff(find(ind));
    data0 = data(ind);
    data0 = data0(:);
    datat = data0(2:end);
    datat1 = data0(1:end-1);
    X = AR1Design(ind,:);
    XB = X*beta_params(:);
    XBlagged = XB(2:end);
    XB = XB(1:end-1);
    
    B = theta(2).^(1:max(lag0))';
    C = (1-B)./(1-theta(2));
    D = theta(1)*C;
    mu = D(lag0)+XB+B(lag0).*(datat1-XBlagged);
    B2 = theta(2).^(2*(1:max(lag0)))';
    C2 = (1-B2)./(1-theta(2)^2);
    s = C2(lag0)*theta(3);
    pb = -0.5*log(s) - ((datat-mu).^2)./(2*s);
    
    ll = sum(pb);
    
    B = can(2).^(1:max(lag0))';
    C = (1-B)./(1-can(2));
    mu = can(1)*C(lag0)+XB+B(lag0).*(datat1-XBlagged);
    B2 = can(2).^(2*(1:max(lag0)))';
    C2 = (1-B2)./(1-can(2)^2);
    s = C2(lag0)*can(3);
    
    pbcan = -0.5*log(s) - ((datat-mu).^2)./(2*s);
    
    llcan = sum(pbcan);
    
    difflikes = llcan - ll;
end
end

function [can] = beta_sampler(params,data,R,sigma,n_AR,Priors, AR1Design, curr_beta, isWaveletModel)

if isWaveletModel{1}
    R = wextend(1,'sym',wextend(1,'sym',R,isWaveletModel{4},'l'),isWaveletModel{5},'r');
    data = isWaveletModel{6};
    AR1Design = isWaveletModel{2};
else
    
end
l = length(R);
ARMU = zeros(l,1);
S = ones(l,1);
X = zeros(size(AR1Design)); 
for r = 0:n_AR-1
    theta = params(1,(r)*3+1:(r+1)*3);
    ind = R==r;
    first = find(ind,1);
    last = find(ind,1,'last');
    lag0 = diff(find(ind));
    data0 = data(ind);
    data0 = data0(:);
    datat = data0(2:end);
    datat1 = data0(1:end-1);
    indt = ind; 
    indt1 = ind;
    indt(first) = false; 
    indt1(last) = false;
    
    B = theta(2).^(1:max(lag0))';
    C = (1-B)./(1-theta(2));
    D = theta(1)*C;
    E = D(lag0)+B(lag0).*datat1;
    
    ARMU(indt) = datat-E; % P(t)-\phi^k P(t-k)
    B2 = theta(2).^(2*(1:max(lag0)))';
    C2 = (1-B2)./(1-theta(2)^2);
    s = C2(lag0)*theta(3);
    S(indt) = s;
    
    TempDesign = AR1Design(ind,:);
    X(indt,:) = TempDesign(2:end,:) - B(lag0).*TempDesign(1:end-1,:); % W_t - \phi^k W_{t-k}
end

if isWaveletModel{1}
    idx = true(l,1); % AR data
    idx([1:isWaveletModel{4},l+isWaveletModel{4}+1:end]) = false; % AR data small
    S2 = 1./(2*S(:)); 
    Xsmall = X(idx,:);
    ARMUsmall = ARMU(idx);
    S2small = S2(idx);
    
    mu = Xsmall*curr_beta(:); 
else
    S2 = 1./(2*S(:));
    mu = X*curr_beta(:); 
end

can = curr_beta;
for m = 1:length(curr_beta)
    r1 = rand;
    r2 = rand;
    can(m) = can(m) + sqrt(-2*log(r1))*cos(2*pi*r2)*sigma(m); 
    
    if isWaveletModel{1}
        if m < isWaveletModel{3} % wavelet model, week indicator parameters
            mu1 = mu; %Xsmall*curr_beta(:);
            pb = - ((ARMUsmall-mu1).^2).*S2small; 
            
            mu2 = mu1 + Xsmall(:,m)*(can(m)-curr_beta(m)); 
            pbcan = - ((ARMUsmall-mu2).^2).*S2small;
        else % wavavelet model, wavelet parameters
            if m == isWaveletModel{3}
                mu = X*curr_beta(:);
            end
            mu1 = mu; 
            pb = - ((ARMU-mu1).^2).*S2;
            
            mu2 = mu1 + X(:,m)*(can(m)-curr_beta(m)); %X*can(:);
            pbcan = - ((ARMU-mu2).^2).*S2;
        end
    else % not a wavelet model, no need for padding
        mu1 = mu; %X*curr_beta(:);
        pb = - ((ARMU-mu1).^2).*S2;
        
        mu2 = mu1 + X(:,m)*(can(m)-curr_beta(m));
        pbcan = - ((ARMU-mu2).^2).*S2;
    end
    
    ll = sum(pb);
    llcan = sum(pbcan);
    
    difflikes = llcan - ll;
    prior = Priors{m};
    testNo = difflikes + log(prior(can(m))) - log(prior(curr_beta(m))); 
    if log(rand) < testNo
        curr_beta(m) = can(m);
        mu = mu2;
    else
        can(m) = curr_beta(m);
    end
end
end

function [ theta_new, acc ] = IR_sampler( theta_old, data, R, sigma, regime, IR_model, qs, ParamPriors ,n_AR)
%IR_sampler returns a MH sample from the posterior of the parameters for an
%independent regime in a MRS model
%   theta_old is a vector of the current samples of the MCMC chain
%   data is a vector of observed prices
%   R is the regime sequence
%   sigma is a vector of length 3 that specifies the proposal variances
%   theta_new is the new position of the MCMC chain
%   the priors are improper and flat.
%   Regime specifies the index of the IR regime that we are sampling from

can = theta_old;
acc = zeros(size(theta_old));

% propose
r1 = rand;
r2 = rand;
% normrnd(theta_old,sigma);
can(1) = sqrt(-2*log(r1))*sin(2*pi*r2)*sigma(1) + theta_old(1);

% accept/reject
difflikes = diff_loglike_ind( data,R,theta_old,can, regime, IR_model, qs, n_AR );
alpha = difflikes;
prior = ParamPriors{1};
if log(rand) < alpha + log(prior(can(1))) - log(prior(theta_old(1)))
    theta_old(1) = can(1);
    acc(1) = 1;
else
    can(1) = theta_old(1);
end

% propose
can(2) = sqrt(-2*log(r1))*cos(2*pi*r2)*sigma(2) + theta_old(2);

% accept/reject
difflikes = diff_loglike_ind( data,R,theta_old,can, regime, IR_model, qs,n_AR );
alpha = difflikes;
prior = ParamPriors{2};
if log(rand) < alpha + log(prior(can(2))) - log(prior(theta_old(2)))
    theta_old(2) = can(2);
    acc(2) = 1;
end

theta_new = theta_old;

end

function [ difflikes ] = diff_loglike_ind( data,R,theta,can,regime,IR_model, qs,n_AR )
% calcualtes canloglike - loglike when only the parameters of the spike
% regime change
% data are the observed prices.
% theta are the old parameters.
% can are the new parameters.
% regime specified which regime corresponds to an AR regime.

if theta(2) <= 0 || can(2) <= 0
    difflikes = -Inf;
else
    data1 = data(R==regime); % the spikes
    llcan = -Inf;
    ll = -Inf;
    if ~isempty(data1)
        switch IR_model
            case 'G'
                ps = - 0.5*log(2*pi*theta(2)) - ...
                    (data1-theta(1)).^2./(2*theta(2));
                pscan = - 0.5*log(2*pi*can(2)) - (data1-can(1)).^2./(2*can(2));
            case 'LN'
                if any(data1) <= 0
                    ps = -Inf;
                    pscan = -Inf;
                else
                    ps = -log( data1*sqrt(2*pi*theta(2)) )-...
                        ( log(data1)-theta(1) ).^2 ./( 2*theta(2) );
                    pscan = -log( data1*sqrt(2*pi*can(2)) )-...
                        ( log(data1)-can(1) ).^2 ./( 2*can(2) );
                end
            case 'LN shifted and reversed'
                q1 = qs(regime+1-n_AR);
                if any(q1-data1) <= 0
                    ps = -Inf;
                    pscan = -Inf;
                else
                    ps = -log( (q1-data1) *sqrt(2*pi*theta(2)) )-...
                        ( log((q1-data1))-theta(1) ).^2 ./( 2*theta(2) );
                    pscan = -log( (q1-data1)*sqrt(2*pi*can(2)) )-...
                        ( log((q1-data1))-can(1) ).^2 ./( 2*can(2) );
                end
            case 'LN shifted'
                q2 = qs(regime+1-n_AR);
                if any(data1-q2) <= 0
                    ps = -Inf;
                    pscan = -Inf;
                else
                    ps = -log( (-q2+data1)*sqrt(2*pi*theta(2)) )-...
                        ( log((-q2+data1))-theta(1) ).^2 ./( 2*theta(2) );
                    pscan = -log( (-q2+data1)*sqrt(2*pi*can(2)) )-...
                        ( log((-q2+data1))-can(1) ).^2 ./( 2*can(2) );
                end
            case 'Gamma'
                q2 = qs(regime+1-n_AR);
                if theta(1) <= 0 || can(1) <= 0 || any(data1-q2) <=0
                    ps = -Inf;
                    pscan = -Inf;
                else
                    ps = -log(theta(2)^theta(1)) ...
                        -gammaln(theta(1))...
                        +(theta(1)-1)*log(data1-q2)...
                        -(data1-q2)/theta(2);
                    pscan = -log(can(2)^can(1)) ...
                        -gammaln(can(1))...
                        +(can(1)-1)*log(data1-q2)...
                        -(data1-q2)/can(2);
                end
        end
        llcan = sum(pscan);
        ll = sum(ps);
        
    end
    difflikes = llcan - ll;
end
end

function [xout] = transform_AR2N(x, x_1, lag, params,index,sorted, AR1Design, beta_params)
% Calculates rediduals for the partially observed AR(1) process
xout = nan(length(x),1);
for n = 1:length(x)
    data = x(n);
    lag_size = lag(n);
    lagged_data = x_1(n);
    B = params(2).^(lag_size);
    C = (1-B)./(1-params(2)); C = C(:);
    mu = params(1)*C+AR1Design(index(n+1),:)*beta_params(:)+params(2).^lag_size.*(lagged_data-AR1Design(index(n),:)*beta_params(:));
    B2 = params(2).^(2*lag_size);
    C2 = (1-B2)./(1-params(2)^2); C2 = C2(:);
    s = C2*params(3);
    xout(n) = (data-mu)./(sqrt(s));
end
if sorted == true
    xout = sort(xout);
else
end
end

function [] = checking_plots(data,param_samps,R_samps,n_AR,n_IR,model, AR1Design, beta_samps)
% produces checking plots for our model
global n_regimes
s3 = size(param_samps,3);
s2 = size(param_samps,2);
L = size(param_samps,1);
Param_samps = nan(s3*L,s2);
for i = 1:size(param_samps,3)
    Param_samps((1:L) + L*(i-1),:) = param_samps(:,:,i);
    beta_params((1:L) + L*(i-1),:,:) = beta_samps(:,:,i);
end
param_samps = Param_samps;
%for k = 1:3
%    h = figure(k);
%    set(h,'Position',[1000,1000,1000,1000])
%end
L = L*s3;
for l = 1:L
    params = param_samps(l,:);
    R = R_samps(:,l);
    for i = 0:(n_AR-1)
        ind = (3)*i + 1; index = ind:(ind+2);
        regime_params = params(index);
        dat = data(R==i);
        if length(dat)>10
            f = find(R==i);
            
            %figure(1) % qq-plots
            h = figure;%subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            qq_plots(dat,model{1}{i+1},diff(f),regime_params,f,AR1Design, beta_params(l,:)); hold on
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            %set(h,'pos',p,'fontsize',6);
            set(gca,'fontsize',16);
            box on
            
            Title = ['QQ plot, ', model{1}{i+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex')
            xlabel('')
                        
            resid = transform_AR2N(dat(2:end), dat(1:end-1), diff(f), regime_params,f,false, AR1Design, beta_params(l,:));
            %figure(2) %residual vs. time plot
            h = figure; %subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            plot(f(2:end),resid,'o'); hold on,
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            %set(h,'pos',p,'fontsize',6);
            set(gca,'fontsize',16);
            box on
            title(''), xlabel('\(t\)','interpreter','latex'), ylabel('residual','interpreter','latex')
            %if l == 1
            Title = ['Residuals vs. time, ', model{1}{i+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex')
            %end
            axis([-inf,inf,-inf,inf])
            
            %figure(3) %residual vs. X(t-1)
            h = figure; %subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            plot(abs(dat(1:end-1)),resid,'o'); hold on,
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            %set(h,'pos',p,'fontsize',6);
            set(gca,'fontsize',16);
            box on
            title(''), xlabel('\(|X_{t-1}|\)','interpreter','latex'), ylabel('residual','interpreter','latex')
            %if l == 1
            Title = ['Residuals vs. \(X_{t-1}\), ', model{1}{i+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex')
            %end
            axis([-inf,inf,-inf,inf])
            
            % figure(4) % Plomb-Scargle periodogram for missing data
            h = figure; 
            plomb(resid,f(2:end));
            title(['Periodogram of residuals, ', model{1}{i+1}, ', Regime ',num2str(i+1)],'interpreter','latex')
            set(gca,'fontsize',16);
            xlabel('Frequence (mHz)','interpreter','latex')
            ylabel('Power/frequency (dB/Hz)','interpreter','latex')
            
            % figuer(5) % ACF
            h = figure; 
            resid_w_nan = nan(size(data)); resid_w_nan(f(2:end)) = resid;
            [a,b] = nanautocorr(resid_w_nan,20,1); hold on,
            stem(0:20,a), hold on, plot(0:20,b*ones(21,1),'r'), plot(0:20,-b*ones(21,1),'r')
            title(['ACF of residuals, ', model{1}{i+1}, ', Regime ',num2str(i+1)],'interpreter','latex')
            axis([0,20,-1,1])
            set(gca,'fontsize',16);
            
            % figuer(6) % PACF
            h = figure; 
            [a,b] = nanparcorr(resid,20,0); hold on,
            stem(0:20,a), hold on, plot(0:20,b*ones(21,1),'r'), plot(0:20,-b*ones(21,1),'r')
            title(['PACF of residuals, ', model{1}{i+1}, ', Regime ',num2str(i+1)],'interpreter','latex')
            axis([0,20,-1,1])
            set(gca,'fontsize',16);
            
        end
    end
    cum_has_q = 0;
    for i = n_AR:(n_regimes-1)
        has_q = strcmp(model{2}{i-n_AR+1},'LN shifted') + strcmp(model{2}{i-n_AR+1},'LN shifted and reversed') + strcmp(model{2}{i-n_AR+1},'Gamma');
        ind = (3)*n_AR + (i-n_AR)*2+cum_has_q + 1; index = ind:(ind+1+has_q);
        cum_has_q = cum_has_q + has_q;
        regime_params = params(index);
        if has_q ~= 1
            dat = data(R==i);
        else
            if strcmp(model{2}{i-n_AR+1},'LN shifted')
                dat = data(R==i)-regime_params(1);
            elseif strcmp(model{2}{i-n_AR+1},'Gamma')
                dat = data(R==i)-regime_params(1);
            elseif strcmp(model{2}{i-n_AR+1},'LN shifted and reversed')
                dat = regime_params(1)-data(R==i);
            end
        end
        if length(dat)>10
            f = find(R==i);
            
            %figure(1) % qq-plots
            h = figure; %subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            qq_plots(dat,model{2}{i-n_AR+1},[],regime_params(2:3),f,[], []), hold on
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            set(gca,'fontsize',16);
            box on
            %if l == 1
            Title = ['QQ plot, ', model{2}{i-n_AR+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex'), xlabel('')
            %end
            
            %figure(2) %Raw residuals
            h = figure; %subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            plot(f,dat,'o'); hold on,
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            set(gca,'fontsize',16);
            box on
            title(''), xlabel('\(t\)','interpreter','latex'), ylabel('residual','interpreter','latex')
            %if l == 1
            Title = ['Residuals vs. time, ', model{2}{i-n_AR+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex')
            %end
            axis([-inf,inf,-inf,inf])
            
            %figure(3) %residual vs. |X(t-1)|
            h = figure; %subplot(L,n_regimes,i+1+(l-1)*n_regimes);
            plot(abs(dat(1:end-1)),dat(2:end),'o'); hold on,
            %p = get(h,'pos');
            %p = p + [-0.015,0,0,0.025-0.02];
            set(gca,'fontsize',16);
            box on
            title(''), xlabel('\(|X_{t-1}|\)','interpreter','latex'), ylabel('residual','interpreter','latex')
            %if l == 1
            Title = ['Residuals vs. \(X_{t-1}\), ', model{2}{i-n_AR+1}, ', Regime ',num2str(i+1)];
            hold on, title(Title,'interpreter','latex')
            %end
            axis([-inf,inf,-inf,inf])
        end
    end
end
end

function [] = qq_plots(x,model,lags,params,times,AR1Design,beta_params);
% makes qq plots
switch model
    case 'G'
        qqplot(x)
    case 'LN'
        pd = makedist('Lognormal','mu',params(1),'sigma',params(2)^0.5);
        qqplot(x,pd)
    case 'LN shifted'
        pd = makedist('Lognormal','mu',params(1),'sigma',params(2)^0.5);
        qqplot(x,pd)
    case 'LN shifted and reversed'
        pd = makedist('Lognormal','mu',params(1),'sigma',params(2)^0.5);
        qqplot(x,pd)
    case 'Gamma'
        pd = makedist('Gamma','a',params(1),'b',params(2));
        qqplot(x,pd)
    case 'AR(1)'
        x = transform_AR2N(x(2:end),x(1:end-1),lags,params,times,true,AR1Design,beta_params);
        qqplot(x)
end
title(''),ylabel(''),axis([-Inf,Inf,-Inf,Inf])
end


