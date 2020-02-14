function [ MODE, MEAN, MEDIAN, BMODE, BMEAN, BMEDIAN ] = mcmc_plot( theta, BetaParams, model, burn_in, DoBetaPlots )
%mcmc_plot makes a trace plot and a ksdensity() plot for each column in
%theta where theta is an mcmc chain.
% mode is the mode of the ksdensity
% mean is the mean of the samples
% median is the median of the samples
s = size(theta);
l = s(2);
n = s(1);
N = size(theta,3);
mcmc = theta;
cum_n_q = 0;
colours = {'b' 'g' 'c' 'm'};
MEAN = [];
MODE = [];
MEDIAN = [];
titles = {'\(q_' '\(\mu_' '\(\sigma_'};
titles2 = {'\(\alpha\)' '\(\phi\)' '\(\sigma\)'};
% plots for parameters
for m = 1:2
    for k = 1:length(model{m})
        if m == 1
            theta = mcmc(:,(1:3) + (k-1)*3,:); l = 3;
        else
            this_model_q = 0;
            if strcmp(model{m}{k},'LN shifted') || strcmp(model{m}{k},'LN shifted and reversed') || strcmp(model{m}{k},'Gamma')
                this_model_q = 1; l = 3;
            else
                l = 2;
            end
            theta = mcmc(:,3*length(model{1})+1+(k-1)*2 + cum_n_q :  3*length(model{1})+1+(k-1)*2+cum_n_q + 2 + this_model_q,:);
            cum_n_q = cum_n_q + this_model_q;
        end
        if nargin < 3 || isempty(burn_in) || burn_in == n;
            figure('Position',[100,100,1049,895])
            
            for i = 1:l
                subplot(l*2+l-1,1,[1,2] + 3*(i-1));
                hold on
                for j = 1:N;
                    hold on
                    plot(1:n,theta(:,i,j),colours{j});
                end
                box on;
                if m ~= 1
                    if l == 2
                        title(strcat(titles(i+1),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                    else
                        title(strcat(titles(i),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                    end
                else
                    title(titles2(i), 'interpreter','latex')
                end
                set(gca,'fontsize',14)
            end
        if m==1
            suptitle(strcat(model{m}{k}, ' #', num2str(k)))
        else
            suptitle(strcat(model{m}{k}, ' #', num2str(k+length(model{1}))))
        end
            burn_in = input('Please enter the number of burn in iterations, burn_in = ')
            close
        end
        
        figure('Position',[100,100,1049,895])
        for i = 1:l
            subplot(l*2+l-1,2,[1,3] + 6*(i-1));
            for j = 1:N
                hold on
                plot(1:burn_in,theta(1:burn_in,i,j),'k');
                plot((burn_in+1):n,theta((burn_in+1):end,i,j),colours{j});
            end
            box on
            if m ~= 1
                if l == 2
                    title(strcat(titles(i+1),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                else
                    title(strcat(titles(i),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                end
            else
                title(titles2(i), 'interpreter','latex')
            end
            set(gca,'fontsize',14)
        end
        
        Mode = zeros(l,1);
        Mean = zeros(l,1);
        Median = zeros(l,1);
        
        for i = 1:l
            
            subplot(l*2+l-1,2,[2,4] + 6*(i-1)); hold on;
            samps = theta(burn_in+1:end,i,:);
            samps = samps(:);
            [a,b] = ksdensity(samps);
            
            ind = max(a)==a;  
            Mode(i) = b(find(ind==1,1));
            Mean(i) = mean(samps);
            Median(i) = median(samps);
            hold on;
            plot(b,a)
            box on
            if m ~= 1
                if l == 2
                    title(strcat(titles(i+1),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                else
                    title(strcat(titles(i),num2str(k+length(model{1})),'\)'), 'interpreter','latex')
                end
            else
                title(titles2(i), 'interpreter','latex')
            end
            set(gca,'fontsize',14)
        end
        if m==1
            suptitle(strcat(model{m}{k}, ' #', num2str(k)))
        else
            suptitle(strcat(model{m}{k}, ' #', num2str(k)+length(model{1})))
        end
        MODE = [MODE;Mode];
        MEAN = [MEAN;Mean];
        MEDIAN = [MEDIAN;Median];
    end
end

% plots for switching
theta = mcmc(:,3*length(model{1})+1+(k)*2 + cum_n_q :  end,:);
l = size(theta,2);
if nargin < 3 || isempty(burn_in) || burn_in == n;
    figure('Position',[100,100,1049,895])
    for i = 1:l
        subplot(sqrt(l),sqrt(l),floor((i-1)/4)+1+4*mod((i-1),4));
        hold on
        for j = 1:N;
            hold on
            plot(1:n,theta(:,i,j),colours{j});
        end
        box on;
        title(['\(p_{',num2str(mod(i-1,sqrt(l))+1),num2str(floor((i-1)/sqrt(l))+1),'}\)'], 'interpreter','latex')
        set(gca,'fontsize',14)
    end
    suptitle('Switching probs - Trace')
    burn_in = input('Please enter the number of burn in iterations, burn_in = ')
    close
end

figure('Position',[100,100,1049,895])
for i = 1:l
    subplot(sqrt(l),sqrt(l),floor((i-1)/sqrt(l))+1+sqrt(l)*mod((i-1),sqrt(l)));
    for j = 1:N
        hold on
        plot(1:burn_in,theta(1:burn_in,i,j),'k');
        plot((burn_in+1):n,theta((burn_in+1):end,i,j),colours{j});
    end
    box on
    title(['\(p_{',num2str(mod(i-1,sqrt(l))+1),num2str(floor((i-1)/sqrt(l))+1),'}\)'], 'interpreter','latex')
    set(gca,'fontsize',14)
end
suptitle('Switching probs - Trace')

Mode = zeros(l,1);
Mean = zeros(l,1);
Median = zeros(l,1);
figure('Position',[100,100,1049,895])
for i = 1:l
    subplot(sqrt(l),sqrt(l),floor((i-1)/sqrt(l))+1+sqrt(l)*mod((i-1),sqrt(l))); hold on;
    samps = theta(burn_in+1:end,i,:);
    samps = samps(:);
    [a,b] = ksdensity(samps);
    ks_x = b; ks_y = a;
    
    ind = max(a)==a;
    Mode(i) = b(ind);
    Mean(i) = mean(samps);
    Median(i) = median(samps);
    hold on;
    plot(b,a)
    box on
    title(['\(p_{',num2str(mod(i-1,sqrt(l))+1),num2str(floor((i-1)/sqrt(l))+1),'}\)'], 'interpreter','latex')
    set(gca,'fontsize',14)
end
suptitle('Switching probs - KS density')

MODE = [MODE;Mode];
MEAN = [MEAN;Mean];
MEDIAN = [MEDIAN;Median];

% plots for trend parameters
if ~isempty(BetaParams)
    Beta = []; 
    for l = 1:size(BetaParams,3)
        Beta = [Beta;BetaParams(:,:,l)];
    end
    if DoBetaPlots
        figure('Position',[100,100,1049,895])
    end
    for ll = 1:size(BetaParams,2)
        if DoBetaPlots
            subplot(size(BetaParams,2),2,ll*2-1)
            hold on
            for c = 1:size(BetaParams,3)
                plot(1:burn_in,BetaParams(1:burn_in,ll,c),'k');
                plot((burn_in+1):n,BetaParams((burn_in+1):end,ll,c),colours{j});
            end
            subplot(size(BetaParams,2),2,ll*2)
            hold on
            ksdensity(Beta(:,ll))
        end
        [a,b] = ksdensity(Beta(:,ll));
        ind = find(max(a)==a,1); 
        BMODE(ll) = b(ind);
    end
    BMEAN = mean(Beta);
    BMEDIAN = median(Beta);
else
    BMODE = [];
    BMEAN = [];
    BMEDIAN = [];
end

end

