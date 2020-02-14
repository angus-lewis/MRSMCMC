function [h] = shftandadd(x,shift)

x = x(:); 
h = nan(size(x)); 
if shift<0
    g = [x;zeros(-2*shift,1)];
    h(1) = sum(g(1:(-2*shift+1)));
    h(2:end) = g(-2*shift+2:end);
else
    g = [zeros(2*shift,1);x];
    h(end) = sum(g(end-2*shift:end));
    h(1:end-1) = g(1:end-2*shift-1);
end
end

