a=10;
X=[-0.2,-0.1,0,1];
fX = (1+sign(X))./2.*X - a*(1-sign(X))./2.*X;
fX(1)=fX(2);

delta = 0.1;

x = 0.28; path_x=[x];
fx = (1+sign(x))./2.*x - a*(1-sign(x))./2.*x; path_fx=[fx];
for i=1:4
    der_x = 1 - a*(1-sign(x))./2;
    x = x - delta * der_x;
    fx = (1+sign(x))./2.*x - a*(1-sign(x))./2.*x;    
    path_x = [path_x;x];
    path_fx = [path_fx;fx];
end
    
plot(X,fX,'LineWidth',2)
hold on
plot_dir(path_x,path_fx)
xlabel('x')
legend({'f(x)','path of GD'},'Location','southeast','FontSize',12)

