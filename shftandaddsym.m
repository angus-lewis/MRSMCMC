function [h] = shftandaddsym(x,shift)

x = x(:); 
if shift<0
    g = [x;zeros(-2*shift,1)];
    h = g(-2*shift+1:end);
    h(1:-2*shift) = h(1:-2*shift) + g(-2*shift:-1:1);
else
    g = [zeros(2*shift,1);x];
    h = g(1:end-2*shift);
    h(end:-1:end-2*shift+1) = g(end-2*shift+1:end) + h(end:-1:end-2*shift+1);
end

end

