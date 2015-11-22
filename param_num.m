function y = param_num(x)
y=0;
for i=1:length(x)-1
    y=y+x(i)*x(i+1);
end
