function [h] = shft(x,shift)

x = x(:); 
h = nan(size(x)); 
if shift<0
    g = [x;zeros(-2*shift,1)];
    h(1:end) = g(-2*shift+1:end);
else
    g = [zeros(2*shift,1);x];
    h(1:end) = g(1:end-2*shift);
end
end
