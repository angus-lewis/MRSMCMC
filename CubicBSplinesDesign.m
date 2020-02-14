function [X,BFun] = CubicBSplinesDesign(Times,NKnots)
%Constructs a design matrix for cubic B-splines

Times = Times(:); 
SortTimes = sort(Times); 
L = length(Times);

M = 4; 
h = 1/(NKnots-1);
q = h:h:(1-h);
Knots = [quantile(Times,0)-1;quantile(Times,q)';quantile(Times,1)+1]; 
E0 = Knots(1); 
EK = Knots(NKnots); 
Tau = [E0*ones(M-1,1);Knots;EK*ones(M-1,1)];

BFun = cell(NKnots+2*M-1,M);
for i = 1:NKnots-2+2*M-1
    BFun{i,1} = @(x) (Tau(i)<=x) .* (x<Tau(i+1)) .* (Tau(i)~=Tau(i+1));
end

for m = 2:M
    for i = 1:NKnots-2+2*M-m
        if Tau(i+m-1)~=Tau(i) 
            f1 = @(x) (x-Tau(i))/(Tau(i+m-1)-Tau(i)).*BFun{i,m-1}(x);
        else
            f1 = @(x) 0;
        end
        if Tau(i+m)~=Tau(i+1)
            f2 = @(x) (Tau(i+m)-x)/(Tau(i+m)-Tau(i+1)).*BFun{i+1,m-1}(x);
        else 
            f2 = @(x) 0; 
        end
        BFun{i,m} = @(x) f1(x) + f2(x);
    end
end 

X = nan(L,NKnots); 
for k = 1:NKnots+M-2
    X(:,k) = BFun{k,M}(Times);
end

end

