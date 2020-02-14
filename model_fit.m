% model_fit
clear all 

%%%% READ IN THE MODEL
T = readtable('jobs.txt','ReadVariableNames',false);
%task = 1%str2num(getenv('task'));
task = str2num(getenv('task'));
MODEL = table2array(T(task,:));  % {'1' 'AR(1)' '2' 'LN_shifted' 'LN_shifted' 'Wavelet' '8' '13'}; %
MODEL = regexp(MODEL,' ','split');
MODEL = MODEL{1}
NAR = str2double(MODEL{1});
ARMODEL = {MODEL{2:NAR+1}};
NIR = str2double(MODEL{NAR + 2});
IRMODEL = {MODEL{NAR+3:NAR+2+NIR}};
for i = 1:NIR
    IRMODEL{i}(find(IRMODEL{i}=='_')) = ' ';
end
TRENDMODEL = MODEL(NAR+2+NIR+1:end);
model = {{ARMODEL{:}},{IRMODEL{:}}};

%%%% READ IN THE DATA
%readdat = readtable('prices.csv');
readdat = readtable('~/fastdir/HPC/prices.csv');
RRP = table2array(readdat(:,5)); RRP = RRP(1:end)';
for i = 1:(length(RRP)/48)
   dat(i) = mean(RRP((i-1)*48+1:i*48));
end
RRP = dat;

% dwtmode('sym')
% [C,L] = wavedec(dat,8,['db','6']); 
% TRENDS = wrcoef('a',C,L,['db','6']);
% [X,S] = SIM_MRS(length(dat),{model{1},{'LN'}},[0,0.9,2,0,1,2,0.9,0.1,0.6,0.4],[1,0]);
% dat = TRENDS(:)'+X(:)'; 
% RRP = dat; 

%%%% Construct the design matrix of the trend model
EYE7 = eye(7);
s = 10*sqrt(var(dat));
if strcmp(TRENDMODEL{1},'BSplines')
    % CUBIC SPLINES
    NKnots = str2double(TRENDMODEL{2});
    C = CubicBSplinesDesign(1:length(RRP),NKnots);
    AR1Design = [[repmat(EYE7,floor(length(RRP)/7),1);EYE7(1:length(RRP)-floor(length(RRP)/7)*7,:)],...
                 C];
	LSE = (C'*C)^-1*C'*dat(:);
    BetaPriors = repmat({@(x)normpdf(x,0,s)},size(AR1Design,2),1); % beta ~ N(,)
    isWaveletModel = {false};
else
    % WAVELET BASIS
    JLEVEL = str2double(TRENDMODEL{2});
    DBTYPE = TRENDMODEL{3};
    [~,~,~,~,W,b,a] = waveletDesign(DBTYPE,JLEVEL,length(RRP)); 
    extl = size(W,1);
    Msmall = [repmat(EYE7,floor(length(dat)/7),1),;EYE7(1:length(dat)-floor(length(dat)/7)*7,:)];
    MBig = wextend('ar','sym',wextend('ar','sym',Msmall,a,'d'),b,'u');
    AR1Design = [MBig,...
             W];
    BetaPriors = repmat({@(x)normpdf(x,0,s)},size(AR1Design,2),1); % beta ~ N(0,10*s)
    isWaveletModel = {true,AR1Design,sum(sum(EYE7)),b,a,...
        wextend(1,'sym',wextend(1,'sym',dat,b,'l'),a,'r')};
    AR1Design = AR1Design(b+(1:length(RRP)),:);
end

%%%% ASSIGN PRIORS TO PARAMETERS
s = 10*sqrt(var(dat)); 
ARPriors = repmat({0, @(x)1, @(x) (1/x)*(x<10*s)*(x>1)},1,NAR); % \phi ~ U[-1,1], log(sigma^2) ~ U[1,10*s]
IRPriors = [];
c = 0;
for i = 1:NIR
    switch model{2}{i}
        case 'LN shifted'
		if c == 0
            temp = {@(x)(x>quantile(RRP,0.66))*(x<quantile(RRP,0.99)),... shifting parameter, q ~ U[0.66 quatile, 0.99 quantile]
                @(x)normpdf(x,0,s),... % mu ~ N(0,s)
                @(x) (1/x)*(x<s)*(x>0.1)}; % log(sigma^2)~U[0.1,s]
            IRPriors = [IRPriors,temp];
		c = c+1; 
		else
		temp = {@(x)(x>quantile(RRP,0.9))*(x<quantile(RRP,0.99)),... shifting parameter, q ~ U[0.66 quatile, 0.99 quantile]
                @(x)normpdf(x,0,s),... % mu ~ N(0,s)
                @(x) (1/x)*(x<s)*(x>0.1)}; % log(sigma^2)~U[0.1,s]
            IRPriors = [IRPriors,temp];
		end
        case 'LN shifted and reversed'
            temp = {@(x)(x>quantile(RRP,0.01))*(x<quantile(RRP,0.33)),... shifting parameter, q ~ U[0.66 quatile, 0.99 quantile]
                @(x)normpdf(x,0,s),... % mu ~ N(0,s)
               @(x) (1/x)*(x<s)*(x>0.1)}; % log(sigma^2)~U[0.1,s]
            IRPriors = [IRPriors,temp];
        case 'Gamma'
		if c == 0
            invgampdf = @(x) (x(1)>0)*x(2)^x(3)*x(1)^(-x(3)-1)*exp(-x(2)/x(1))/gamma(x(3)); % x(1) is data, x(2) is the scale parameter beta, x(3) is the shape parameter, alpha
            temp = {@(x)(x>quantile(RRP,0.66))*(x<quantile(RRP,0.99)),... shifting parameter, q ~ U[0.66 quatile, 0.99 quantile]
                @(x) invgampdf([x-1,5.5,3]),... % mu ~ N(0,s)
                @(x) (1/x)*(x<s)*(x>0.1)}; % log(sigma^2)~U[0.1,s]
            IRPriors = [IRPriors,temp];
		c = c+1;
		else
		temp = {@(x)(x>quantile(RRP,0.9))*(x<quantile(RRP,0.99)),... shifting parameter, q ~ U[0.66 quatile, 0.99 quantile]
                @(x)normpdf(x,0,s),... % mu ~ N(0,s)
                @(x) (1/x)*(x<s)*(x>0.1)}; % log(sigma^2)~U[0.1,s]
            IRPriors = [IRPriors,temp];
		end
    end
end
Priors = {ARPriors{:},IRPriors{:}};

%%%% BEGIN INFERENCE
NSamps = 2000000;
BurnIn = 500000; 
tic
[ParamSamps, RegimeProbs, BetaSamps] = MRS_MCMC_FN(RRP,NSamps,BurnIn,model,randi(2,length(RRP),1)-1,0.1,true,Priors,AR1Design,BetaPriors,isWaveletModel);
toc

%%%% SAVE THINGS
main_title = [MODEL{:}];
main_title = regexp(main_title,'(','split');
main_title = [main_title{:}];
main_title = regexp(main_title,')','split');
main_title = [main_title{:}];
DirName = ['Model',main_title];
mkdir(DirName)
%main_title = ['',DirName,'/'];
main_title = ['~/fastdir/HPC/',DirName,'/'];

%%%% SAVE CHECKING PLOTS
NFigs = length(findobj('type','figure'));
for h = 1:NFigs
    FigHandle = figure(h);
    figuretitle = regexp(FigHandle.CurrentAxes.Title.String,' ','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,',','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,')','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'(','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'\','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'/','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'{','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'}','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'-','split');
    figuretitle = [figuretitle{:}];
    figuretitle = regexp(figuretitle,'\.','split');
    figuretitle = [figuretitle{:}];
    print([main_title,'fig',figuretitle,'figNo',num2str(h)],'-depsc'),
    savefig(FigHandle,[main_title,'fig',figuretitle,'figNo',num2str(h),'.fig']),
end
close all

%%%% MAKE AND SAVE OTHER PLOTS 
[MODE,MEAN,MEDIAN,BMODE,BMEAN,BMEDIAN] = mcmc_plot(ParamSamps,BetaSamps,model,BurnIn,true);
save([main_title,'data'],'*','-v7.3');
NFigs = length(findobj('type','figure'));
for h = 1:NFigs
try
    FigHandle = figure(h);
    print([main_title,'fig','SamplesPlot','figNo',num2str(h)],'-depsc'),
    savefig(FigHandle,[main_title,'fig','SamplesPlot','figNo',num2str(h),'.fig']),
catch ME 
disp('plot save failed')
h
end
end
close all 

trend = AR1Design*BMEAN';
plot(trend)
axis([-inf,inf,-inf,inf])
datetick('x',3)
xlabel('time')
ylabel('$ AUD')
title('Estimated seasonal component - mean')
set(gca,'fontsize',14)
print([main_title,'fig','seasonalest'],'-depsc')
savefig([main_title,'fig','seasonalest','.fig'])
close all
details = dat(:)-trend(:);
plot(details)
axis([-inf,inf,-inf,inf])
datetick('x',3)
xlabel('time')
ylabel('$ AUD')
title('Estimated stochastic component - mean')
set(gca,'fontsize',14)
print([main_title,'fig','stochasticest'],'-depsc')
savefig([main_title,'fig','stochasticest','.fig'])
close all 





