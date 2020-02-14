function [Wsp0,Wzpd,Wsym,WFull,W,b,a] = waveletDesign(waveNo,J,N)

[LoD] = wfilters(['db',waveNo]);
a = str2double(waveNo);
Wsp0 = eye(N); % operator to calculate coefficients with a zeroth order extension
Wzpd = eye(N); % operator to calculate coefficients with zero padding
Wsym = eye(N); % operator to calculate coefficients with a symmetric extension
N2 = 2^(ceil(log2(N+2*a-2)));
L = N;
LN = N;

for j = 1:J
    % Construct the equivalent decomposition operator with a first order
    % extension
    X = zeros(N,ceil(N/2)+a-1);
    h = [LoD(end:-1:1),zeros(1,N-length(LoD))];
    for l = 1:size(X,2)
        X(:,l) = shftandadd(h,l-a);
    end
    Wsp0 = X'*Wsp0;
    % Construct the equivalent decomposition operator with zero padding
    % extension
    for l = 1:size(X,2)
        X(:,l) = shft(h,l-a);
    end
    Wzpd = X'*Wzpd;
    % Construct the equivalent decomposition operator with a symmetric
    % extension
    for l = 1:size(X,2)
        X(:,l) = shftandaddsym(h,l-a);
    end
    Wsym = X'*Wsym;
    N = size(Wzpd,1);
end
Wsp0 = Wsp0';
Wzpd = Wzpd';
Wsym = Wsym';

% Construct the equivalent decomposition operator with a symmetric
% extension. Boy is this code clunky... but it works!!
[LoD,HiD] = wfilters(['db',waveNo]);
WFull = eye(L+2*2*(a-1)+mod(L,2));
currsettings = dwtmode('status','nodisp'); 
dwtmode('zpd','nodisp');

for j = 1:J-1
    C = [];
    for i = 1:L+2*2*(a-1)+mod(L,2)
        x = zeros(L+2*2*(a-1)+mod(L,2),1);
        x(i) = 1;
        temp = x;
        temp = dwt(temp,LoD,HiD);
        C(i,:) = temp;
    end
    C = C(:,a:end-(a-1));
    C = C(:,[2*(a-1):-1:1,1:end,end:-1:end-2*(a-1)+(1-mod(floor((L-1)/2) + a,2))]);
    L = floor((L-1)/2) + a;
    WFull = C'*WFull;
end

if J>1
    l = size(C,2);
else
    l = L+2*2*(a-1)+mod(L,2);
end
C = [];
for i = 1:l
    x = zeros(l,1);
    x(i) = 1;
    temp = x;
    temp = dwt(temp,LoD,HiD);
    C(i,:) = temp;
end
C = C(:,a:end-(a-1)); 
WFull = C'*WFull;

% Hack it!
fl = 2*a; 
for j = 2:J
    fl = fl + 2^(j-1)*(2*a-1);
end
nz = 2^J*(ceil((fl-2^J)/2^J)) - (fl-2^J);

C = [];
for i = 1:fl+nz
    x = zeros(nz+fl,1);
    x(i) = 1;
    temp = x;
    for j = 1:J
        temp = dwt(temp,LoD,HiD);
    end
    F(i) = temp(ceil((fl-2^J)/2^J)+1);
end

F(F==0) = []; F = F(:);

[~,ind] = wavedec(1:LN,J,['db',waveNo]);
X = [];
h = [F;zeros((ind(1)-1)*2^J,1)];
for i = 1:ind(1)
    X(:,i) = circshift(h,(i-1)*2^J);
end
W = X;
b = fl-2^J;
a = size(X,1)-LN-b;

dwtmode(currsettings,'nodisp');
end













