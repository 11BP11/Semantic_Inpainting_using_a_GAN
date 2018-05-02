a=1; b=0.05;
x = linspace(-1,0.1);
y = linspace(-0.1,1);
[X,Y] = meshgrid(x,y);
Z = a*(X+Y).^2 + b*(X-Y).^2;

v=0.2*[0:0.1:1];

delta = 0.5;

x = -0.8; path_x=[x];
y = 0.7; path_y=[y];
for i=1:100
    der_x = a*2*(x+y) + b*2*(x-y);
    der_y = a*2*(x+y) - b*2*(x-y);
    x = x - delta * der_x;
    y = y - delta * der_y;
    path_x = [path_x,x];
    path_y = [path_y,y];
end
    
figure
contour(X,Y,Z,v)
hold on
plot(path_x,path_y,'r')
xlabel('x') 
ylabel('y')

