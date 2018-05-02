a=1; b=0.05;
x = linspace(-1,0.3);
y = linspace(-0.3,1);
[X,Y] = meshgrid(x,y);
Z = a*(X+Y).^2 + b*(X-Y).^2;

vv=0.2*[0:0.1:1];

epsilon = 0.5;
beta1 = 0.9; beta2 = 0.99;
v=[0,0]; w=[0,0];
x = -0.8; path_x=[x]
y = 0.7; path_y=[y]
for t=1:100
    g = [a*2*(x+y) + b*2*(x-y), a*2*(x+y) - b*2*(x-y)];
    v = beta1 * v + (1-beta1) * g;
    w = beta2 * w + (1-beta1) * g.^2;
    v_hat = v / (1-beta1^t);
    w_hat = w / (1-beta2^t);
    delta = -epsilon * v_hat ./ (sqrt(w_hat)+10^(-8));
    x=x+delta(1); y=y+delta(2);
    path_x = [path_x,x];
    path_y = [path_y,y];
end
    
figure
contour(X,Y,Z,vv)
hold on
plot(path_x,path_y,'r')
xlabel('x') 
ylabel('y')

