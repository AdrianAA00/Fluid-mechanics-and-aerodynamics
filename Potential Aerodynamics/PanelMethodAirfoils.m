%-----------------------------------------------------------------------
%                       Panel method for airfoils
%------------------------------------------------------------------------

%% Spanish
%% METODO DE PANELES PARA PERFILES AERODINAMICOS
clear all
format short
c=1;               %cuerda del perfil
a=0; b=c-a;        %puntos inicial y final del esqueleto del perfil
Vinf=1;            %velocidad de la corriente incidente no perturbada  [m/s]
rho=1;             %densidad del fluido
pinf=1;            %presion de la corriente incidente sin perturbar
%   alpha=0;  %angulo de ataque [rad]


%%Elección perfil NACAXYZK
global xfmax fmax tmax;     %parámetros para el perfil NACA de 4 cifras
%Cifras XYZK = 8884,5 
fmax = (1d0 + fix(8/2d0))/100d0;%4/100;%
xfmax =(20d0 + 10d0*fix(8d0/4d0))/100d0; %40/100;%
tmax =(7d0 + fix(84.5/9d0))/100d0;% 12/100;%
 
 
%% NODOS Y PUNTOS DE CONTROL DE LOS PANELES

N = 200; % numero de paneles para modelizar el perfil 
          % debe ser par si queremos que coja el origen

x=zeros(N+1,1);     %nodos
z=zeros(N+1,1);
xc=zeros(N,1);      %puntos de control
zc=zeros(N,1);


%% %DISTRIBUCION EQUIESPACIADA EN LA CUERDA
% % %distribución equiespaciada en la cuerda
for i=1:N/2   %intrados
    c_param(i)= 2*c/N + (c/(N/2))*(-i+(N/2)); 
end
for i=N/2+1:N+1   % extrados
    c_param(i)= (c/(N/2))*(i-(N/2)-1);    
end
%coordenada x nodos si conozco la funcion del perfil
for i=1:N/2       %intrados 
    x(i) = xp_intra(c_param(i));
end
for i=N/2+1:N+1       % extrados
    x(i) = xp_extra(c_param(i));
end
%coordenada z nodos si conozco la funcion del perfil
for i=1:N/2       %intrados 
    z(i) = zp_intra(c_param(i));
end
for i=N/2+1:N+1       % extrados
    z(i) = zp_extra(c_param(i));
end


%% DISTRIBUCION NO EQUIESPACIADA EN LA CUERDA
% %coordenadas x, z de los nodos
% for i = 1:N+1                   
%    c_param = 0.5*(1d0 + cos(2d0*(i-1d0)*pi/N));    %Valores en [1 -> 0 ->1] 
%                                                    %Concentración de ptos en zonas de mayor curvatura                                                  
%    if i < ((N+1)/2d0)
%       x(i) = xp_intra(c_param);
%       z(i) = zp_intra(c_param);
%    end 
%        
%    if i >= ((N+1)/2d0)    
%       x(i) = xp_extra(c_param);
%       z(i) = zp_extra(c_param);
%    end  
% end
% 


%% Coordenadas puntos de control
for i=1:N 
    xc(i)=(x(i+1)+x(i))/2;
    zc(i)=(z(i+1)+z(i))/2;
end


%% PARAMETROS GEOMETRICOS DE LOS PANELES

n=zeros(2,N);t=zeros(2,N);

[minx,minx_pos]=min(x); %para poder definir los angulos correctamente

for i=1:(minx_pos-1)
    theta(i)=pi+atan((z(i+1)-z(i))/(x(i+1)-x(i)));      %angulo del panel
    n(:,i)=[-sin(theta(i));cos(theta(i))];             %normal al panel
    t(:,i)=[cos(theta(i));sin(theta(i))];              %tangente al panel
    long(i)=sqrt((x(i+1)-x(i))^2+(z(i+1)-z(i))^2);   %longitud del panel
end

for i=minx_pos:N
    theta(i)=atan((z(i+1)-z(i))/(x(i+1)-x(i)));      %angulo del panel
    n(:,i)=[-sin(theta(i));cos(theta(i))];             %normal al panel
    t(:,i)=[cos(theta(i));sin(theta(i))];              %tangente al panel
    long(i) = sqrt((x(i+1)-x(i))^2+(z(i+1)-z(i))^2);   %longitud del panel
end

% thetha(:,1) = theta;
% thetha = thetha*180/pi
% longi(:,1) =  long;
% longi = longi

perfil=plot(x,z); %representa el perfil representado por paneles

%% RESOLUCION PASO A PASO DEL PROBLEMA

a1=zeros(N);a2=zeros(N);a3=zeros(N);a4=zeros(N);a5=zeros(N);a6=zeros(N);a7=zeros(N);a8=zeros(N);a9=zeros(N);a10=zeros(N);
A=zeros(N,N+1);B=zeros(N,N+1);C=zeros(N,N+1);b=zeros(N+1,1);
gamma=zeros(N+1,1); %distribucion de torbellinos en los nodos


for j=1:N
    for i=1:N
        %ejes locales del panel
         rij=sqrt((xc(i)-x(j))^2+(zc(i)-z(j))^2);
         rij1=sqrt((xc(i)-x(j+1))^2+(zc(i)-z(j+1))^2);
         xc_loc(i)=(xc(i)-x(j))*cos(theta(j))+(zc(i)-z(j))*sin(theta(j));          %coordenada puntos de control en ejes locales
         zc_loc(i)=(zc(i)-z(j))*cos(theta(j))-(xc(i)-x(j))*sin(theta(j));
        
        a1(i,j)=1/(2*pi)*log(rij1/rij);
        if i==j
            beta(i,j)=pi;
        else    
            if (xc_loc(i)-long(j)<0) && (xc_loc(i)>0)     %por los signos del arcotangente hay que separar los casos
                if (zc_loc>0)
                beta(i,j)=(pi+atan(zc_loc(i)/(xc_loc(i)-long(j)))-atan(zc_loc(i)/xc_loc(i)));
                else
                beta(i,j)=(-pi+atan(zc_loc(i)/(xc_loc(i)-long(j)))-atan(zc_loc(i)/xc_loc(i)));
                end
            else
                beta(i,j)=(atan(zc_loc(i)/(xc_loc(i)-long(j)))-atan(zc_loc(i)/xc_loc(i)));
            end
        end
        a2(i,j)=beta(i,j)/(2*pi);
        a3(i,j)=1/long(j)*(a1(i,j)*zc_loc(i)+a2(i,j)*xc_loc(i));
        a4(i,j)=1/long(j)*(a1(i,j)*xc_loc(i)+long(j)/(2*pi)-a2(i,j)*zc_loc(i));
        a5(i,j)=a2(i,j)-a3(i,j);
        a6(i,j)=a1(i,j)-a4(i,j);
        %cambio a ejes globales
        a7(i,j)=a5(i,j)*cos(theta(j))-a6(i,j)*sin(theta(j));
        a8(i,j)=a3(i,j)*cos(theta(j))-a4(i,j)*sin(theta(j));
        a9(i,j)=a5(i,j)*sin(theta(j))+a6(i,j)*cos(theta(j));
        a10(i,j)=a3(i,j)*sin(theta(j))+a4(i,j)*cos(theta(j));
    end
end


for i=1:N
        B(i,1)=a7(i,1);
        B(i,N+1)=a8(i,N);
    for j=2:N  
        B(i,j)=a8(i,j-1)+a7(i,j);
    end
    
        C(i,1)=a9(i,1);
        C(i,N+1)=a10(i,N);
    for j=2:N  
        C(i,j)=a10(i,j-1)+a9(i,j);
    end
end


for j=1:N+1
    for i=1:N
        A(i,j)=-B(i,j)*sin(theta(i))+C(i,j)*cos(theta(i));
    end
        A(N+1,j)=B(1,j)*cos(theta(1))+C(1,j)*sin(theta(1))+B(N,j)*cos(theta(N))+C(N,j)*sin(theta(N));
end

% cl_alfa=zeros(N,3);cl_extrados_alfa=zeros(N/2,3);cl_intrados_alfa=zeros(N/2,3);
Cl_alfa=zeros(1,3); Cm_alfa=zeros(1,3); Cm_cuarto_alfa=zeros(1,3); xcp_alfa=zeros(1,3);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Caso 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Con esto pintamos gráficas -2 0 y 5   // Desactivar caso 2
% for iter=1:3   
% caso = 1;
% 
%     if(iter==1) 
%         alpha=-2*pi/180;
%     end 
%     
%      if(iter==2) 
%         alpha=0;
%      end 
%      
%      if(iter==3) 
%         alpha=5*pi/180;
%      end 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Caso 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%%Con esto pintamos gráficas en función de alpha // Desactivar caso 1 
for iter=1:301
caso = 2;
alpha=(pi/180)*(-30+2*(iter-1)/10);  %Evaluamos alphas desde -30 a 30 grados
alpha_grados(iter) = (180/pi)*alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Resolver sistema ecuaciones
     
for i=1:N
    b(i)=-Vinf*sin(alpha-theta(i));
end
    b(N+1)=-Vinf*(cos(alpha-theta(1))+cos(alpha-theta(N)));


gamma=A^(-1)*b;     %distribucion de torbellinos en los nodos


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CARACTERISTICAS AERODINAMICAS EN LOS PUNTOS DE CONTROL
v_pert=zeros(N+1,1);Vtg=zeros(N,1);Cp=zeros(N,1);p=zeros(N,1);
l=zeros(N,1);cl=zeros(N,1);Cl=0;m=zeros(N,1);cm=zeros(N,1);Cm=0;cl_extrados=zeros(N/2,1);cl_intrados=zeros(N/2,1);
d=zeros(N,1);delta=zeros(N,1);xc_extrados=zeros(N/2,1);xc_intrados=zeros(N/2,1);Cm_cuarto=0;xcp=0;

%velocidad tangencial en los puntos de control
for i=1:N
    for j=1:N+1
        sum=gamma(j)*(B(i,j)*cos(theta(i))+C(i,j)*sin(theta(i)));
        v_pert(i)=v_pert(i)+sum;    %perturbacion de la velocidad por los torbellinos
    end
    Vtg(i)= Vinf*cos(theta(i)-alpha) +v_pert(i) ; 
end

%coeficiente de presiones y distibucion de presiones
for i=1:N
    Cp(i)=1-(Vtg(i)/Vinf)^2;
    p(i)=1/2*rho*Vinf^2*Cp(i)+pinf;
end

%sustentacion y coeficiente de sustentacion
for i=1:N/2
    l(i)=-(p(i)-pinf)*long(i)*cos(theta(i)-alpha);
    cl(i)=l(i)/(1/2*rho*Vinf^2*long(i));    %coeficiente de sustentacion de cada placa
    cl_intrados(i,1)=cl(i);
    xc_intrados(i,1)=xc(i);
    Cl=Cl+l(i)/(1/2*rho*Vinf^2*c);      %coeficiente de sustentacion total
end
for i=(N/2+1):N
    l(i)=-(p(i)-pinf)*long(i)*cos(theta(i)-alpha);
    cl(i)=l(i)/(1/2*rho*Vinf^2*long(i));    %coeficiente de sustentacion de cada placa
    cl_extrados(i-N/2,1)=cl(i);
    xc_extrados(i-N/2,1)=xc(i);
    Cl=Cl+l(i)/(1/2*rho*Vinf^2*c);      %coeficiente de sustentacion total
end

%momentos y coeficiente de momentos
for i=1:N
    d(i)=sqrt(xc(i)^2+zc(i)^2);     %distancia al origen
    delta(i)=atan(zc(i)/xc(i));     %angulo con el origen
    m(i)=(p(i)-pinf)*long(i)*d(i)*cos(delta(i)-theta(i));  %theta tiene en cuenta cambio extrados intrados
    cm(i)=m(i)/(1/2*rho*Vinf^2*long(i)*c); %coeficiente de momentos de cada placa
    Cm=Cm+m(i)/(1/2*rho*Vinf^2*c^2);
end

    %Momento según donde se tome momentos en puntos de la cuerda
    if (iter == 101)  %alpha=-5
         for i = 1:N+1
             x_momento(i) = (i-1)/N;
             Cm_pto(i,1) = Cm + (i-1)/N* cos(alpha)*Cl;
         end
    end  
    
    if (iter == 151)  %alpha=0
         for i = 1:N
             Cm_pto(i,2) = Cm + (i-1)/N* cos(alpha)*Cl;
         end
    end  
    
    if (iter == 201)  %alpha=5
        for i = 1:N
             Cm_pto(i,3) = Cm + (i-1)/N* cos(alpha)*Cl;
        end
    end

    
    %Momento respecto de un cuarto de la cuerda
    Cm_cuarto=Cm + 0.25/c* cos(alpha)*Cl;
    
    %Momento respecto a un medio de la cuerda
    Cm_medio=Cm + 0.5/c* cos(alpha)*Cl;
    
    %Momento respecto centro aerodinámico
    Cm_a=Cm + 0.270/c* cos(alpha)*Cl;
    
    
    %Posición del centro de presiones
    %Cmba=-Cl*xcp/c Cmba es el coeficiente de momento respecto en borde de
    %ataque en nuestro caso coincide con Cm
    xcp=-c*Cm/Cl;
    
    cl_alfa(:,iter)=cl;
    cl_extrados_alfa(:,iter)=cl_extrados;
    cl_intrados_alfa(:,iter)=cl_intrados;
    
    Cl_alfa(iter)= Cl;
    Cm_alfa(iter)= Cm;
    Cm_cuarto_alfa(iter)= Cm_cuarto;
    Cm_medio_alfa(iter)= Cm_medio;
    Cm_a_alfa(iter) = Cm_a;
    
    xcp_alfa(iter)= xcp;
      
    
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% REPRESENTACIONES GRAFICAS
%% Representar graficamente perfil
m = 1000;  %Ptos representación
x_panel_dibujo=zeros(m,1); z_panel_dibujo=zeros(m,1);

%Pintamos perfil
for i= 1: m
   x_param = 0.5*(1d0 + cos(2d0*(i-1d0)*pi/m));   
    %Valores en [1 -> 0 ->1] 
    %Concentracion de ptos en zonas de mayor curvatura para pintar mejor.
    %Para paneles se usa equiespaciada por mejores resultados
                 
   if i < ((m+1)/2d0)
      x_panel_dibujo(i) = xp_extra(x_param);
      z_panel_dibujo(i) = zp_extra(x_param);
   end 
       
   if i >= ((m+1)/2d0)    
      x_panel_dibujo(i) = xp_intra(x_param);
      z_panel_dibujo(i) = zp_intra(x_param);
   end  
end

if (caso == 1)
figure (1)
t=tiledlayout(2,1);

nexttile
perfil=scatter(x,z); 
hold on 
plot(x,z)
axis equal
hold off
title('Perfil Modelizado')

nexttile
panel_m = plot(x_panel_dibujo,z_panel_dibujo);
title('Curvatura del perfil real')
axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Caso 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gráficas de alpha en -2, 0 y 5
%Gráfica Cl vs xc/c alfa=-2º

 figure(2)
 t=tiledlayout(1,1);
 
 nexttile
 hold on 
 scatter(xc_extrados/c,cl_extrados_alfa(:,1))
 scatter(xc_intrados/c/c,-cl_intrados_alfa(:,1))
 plot(xc_extrados/c,cl_extrados_alfa(:,1),xc_intrados/c, -cl_intrados_alfa(:,1),'LineWidth',2)
 hold off
 xlabel('x/c','Fontsize', 20)
 ylabel('cl','Fontsize', 20)
 legend('extrados','intrados')
 title('cl extrados e intrados vs x/c para alfa -2º','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;
 
 
 %Gráfica Cl vs xc/c alfa=0º

 figure(3)
 t=tiledlayout(1,1);
 
 nexttile
 hold on 
 scatter(xc_extrados/c,cl_extrados_alfa(:,2))
 scatter(xc_intrados/c,-cl_intrados_alfa(:,2))
 plot(xc_extrados/c,cl_extrados_alfa(:,2),xc_intrados/c,-cl_intrados_alfa(:,2),'LineWidth',2)
 hold off
 xlabel('x/c','Fontsize', 20)
 ylabel('cl','Fontsize', 20)
 legend('extrados','intrados')
 title('cl extrados e intrados vs x/c para alfa 0º','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;
 
 
 %Gráfica Cl vs xc/c alfa= 5º

 figure(4)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 scatter(xc_extrados/c,cl_extrados_alfa(:,3))
 scatter(xc_intrados/c,-cl_intrados_alfa(:,3))
 plot(xc_extrados/c,cl_extrados_alfa(:,3),xc_intrados/c,-cl_intrados_alfa(:,3),'LineWidth',2)
 hold off
 xlabel('x/c','Fontsize', 20)
 ylabel('cl','Fontsize', 20)
 legend('extrados','intrados')
 title('cl extrados e intrados vs x/c para alfa 5º','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;
 
 Cl_alfa
 Cm_alfa
 Cm_cuarto_alfa
 xcp_alfa

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Caso 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gráficas Cl, Cm, Cm1/4, xcp en función de alpha
elseif (caso == 2)
%     Cl_alfa(:,iter)= Cl;
%     Cm_alfa(:,iter)= Cm;
%     Cm_cuarto_alfa(:,iter)= Cm_cuarto;
%     Cm_medio_alfa(:,iter)= Cm_medio;
%     
%     xcp_alfa(:,iter)= xcp;
 %Gráfica Cl vs xc/c alfa= 5º

 figure(5)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(alpha_grados,Cl_alfa,'LineWidth',2)
 hold off
 xlabel('alpha [grados]','Fontsize', 20)
 ylabel('Cl','Fontsize', 20)
 legend('Cl')
 title('Cl perfil en función de alpha','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;

 figure(6)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(alpha_grados,Cm_alfa,'LineWidth',2)
 hold off
 xlabel('alpha [grados]','Fontsize', 20)
 ylabel('Cm ba','Fontsize', 20)
 legend('Cm ba')
 title('Cm en el borde de ataque en función de alpha','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;
 
 
 figure(7)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(alpha_grados,Cm_alfa,alpha_grados,Cm_cuarto_alfa, alpha_grados, Cm_medio_alfa,'LineWidth',2)
 hold off
 xlabel('alpha [grados]','Fontsize', 20)
 ylabel('Cm','Fontsize', 20)
 legend('Cm (ba)','Cm (1/4)','Cm (1/2)')
 title('Cm perfil en función de alpha','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;

 figure(8)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(alpha_grados,xcp_alfa,'LineWidth',2)
 hold off
 xlabel('alpha [grados]','Fontsize', 20)
 ylabel('xcp','Fontsize', 20)
 legend('xcp (ba)')
 title('x centro presiones perfil en función de alpha','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;

             
 figure(9)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(x_momento,Cm_pto,'LineWidth',2)
 hold off
 xlabel('x/c momentos','Fontsize', 20)
 ylabel('Cm_x','Fontsize', 20)
 legend('Cm_x -5º', 'Cm_x 0º', 'Cm_x 5º')
 title('Momentos Cm_x en puntos de la cuerda para alpha: -5, 0 , 5','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;     
 
 figure(10)
 t=tiledlayout(1,1);
 
 nexttile
 hold on
 plot(alpha_grados,Cm_alfa,alpha_grados, Cm_a_alfa,'LineWidth',2)
 hold off
 xlabel('alpha [grados]','Fontsize', 20)
 ylabel('Cm','Fontsize', 20)
 legend('Cmba','Cmca')
 title('Cmca y Cmba perfil en función de alpha','Fontsize', 20)
 ax = gca;
 ax.XAxis.FontSize = 15;
 ax.YAxis.FontSize = 15;

end    

%%
% %% TEORÍA POTENCIAL LINEALIZADA (TPL), RESOLUCIÓN MEDIANTE EL MÉTODO DE GLAUERT
% 
% %Problema directo (de análisis), conocemos la geometría del perfil y
% %buscamos la distribución de Cl y Cp sobre el mismo
% %Cambio de variable de Glauert: x=[(b+a)/2]+[(b-a)/2]*cos(t)
% 
% c=1;    %cuerda del perfil
% ap=0; bp=c-ap; %puntos inicial y final del esqueleto del perfil
% %fmax = (1d0 + fix(8/2d0))/100d0;           %ya estan definidas antes
% %xfmax = (20d0 + 10d0*fix(8d0/4d0))/100d0;
% tfmax = acos(2d0*xfmax-1d0);                %por el cambio de variable de glauert
% 
% 
% %Cálculo de los coeficientes de la serie
% syms TH at X
% 
% %la derivada de la curvatura esta hecha analiticamente y se ha hecho el cambio de variable
% %aideal=(1/pi)*integral(d(zc)/dx,0,pi)  ; zc esta definida a trozos
% aideal = double(1/pi*(int( (fmax/(1-xfmax)^2)*(2d0*xfmax-2d0*((bp+ap)/2+(bp-ap)/2*cos(TH))) ,TH,0,tfmax)+int( (fmax/xfmax^2)*(2d0*xfmax-2d0*((bp+ap)/2+(bp-ap)/2*cos(TH))) ,TH,tfmax,pi))); %ángulo de ataque ideal
% A0=at-aideal;
% r = 20; %número de coeficientes de la serie de Fourier
% AF = zeros(1,r);
% %AF_n=-(2/pi)*integral(d(zc)/dx)*cos(n*theta),0,pi)
% for i=1:r
%     AF(i)=-2/pi*(int( (fmax/(1-xfmax)^2)*(2d0*xfmax-2d0*((bp+ap)/2+(bp-ap)/2*cos(TH)))*cos(i*TH) ,TH,0,tfmax)+int( (fmax/xfmax^2)*(2d0*xfmax-2d0*(((bp+ap)/2)+((bp-ap)/2)*cos(TH)))*cos(i*TH) ,TH,tfmax,pi));
% end
% 
% %Velocidad horizontal sobre el perfil uTPL=u(th)/U_inf
% uTPL = zeros;
% for i=1:r
%     uTPL = uTPL + AF(i)*sin(i*TH);
% end
% uTPL = uTPL + A0*tan(TH/2);
% %Volviendo a la varable x (desde la variable theta)
% UTPL = subs(uTPL,TH,acos((X-((bp+ap)/2))/((bp-ap)/2)));
% 
% %Distribución de cp y cl sobre cada punto del perfil
% cpTPL = -2*UTPL;
% clTPL = 4*UTPL;
% 
% %Coeficientes globales
% ClTPL = 2*pi*(A0+AF(1)/2);
% Cma_TPL = -pi/4*(AF(1)+AF(2));       %centro de momentos en c/4 del borde de ataque
% Cl_TPL =double(subs(ClTPL,at,alpha));
% 
% 
% %posicion de centro de presiones respecto del centro aerodinámico
% centro_p=-Cma_TPL*c/ClTPL;
% 
% 
% %Ángulo de sustentación nula
% asn = -aideal+AF(1)/2;
% 
% %Gráficas (añadir lo que se quiera)
% figure(5)
% fplot(subs(clTPL,at,aideal),[0 1])
% figure(6)
% fplot(ClTPL,[-0.5 0.5])
% 
% %% FUNCIONES PARA LA FORMA DEL PERFIL
% % %Forma aleatoria
% % function [z_int]=z_perfil_int(x)        %intrados    
% %     z_int=2*x+3^2-5 ;       %distribucion de alturas en el intrados        
% % end
% % 
% % function [z_ext]=z_perfil_ext(x)        %extrados
% %     z_ext=2*x+3;        %distribucion de alturas en el extrados         
% % end
% 


%% FUNCIÓN PERFIL NACA 4 CIFRAS

%Proporciona la línea de curvatura
function zc_ = zcurv(x)
    global xfmax fmax
 
    if (x >= 0d0) && (x <= xfmax)
        zc_ = (fmax/xfmax^2)*(2d0*xfmax*x-x^2);
    elseif (x <= 1.0001) && (x >= xfmax)
        zc_ = (fmax/(1-xfmax)^2)*((1-2d0*xfmax)+2d0*xfmax*x-x^2);
    else
        disp('Valor x/c entre 0 y 1')
        quit
    end
end

%Proporciona la derivada de la linea de curvatura para propagar espesor ortogonalmente
function zc_I_ = zc_I(x)
    global xfmax fmax
    
    if (x >= 0d0) && (x <= xfmax)
        zc_I_ = (fmax/xfmax^2)*(2d0*xfmax-2d0*x);
    elseif (x <= 1.0001) && (x >= xfmax)
        zc_I_ = (fmax/(1-xfmax)^2)*(2d0*xfmax-2d0*x);
    end
end

%Proporciona el espesor. Debe ir perpendicular a la linea de curvatura
function  ze_ = ze(x)
    global tmax
    p = 0.2969*sqrt(x)-0.1260*x-0.3516*x^2+0.2843*x^3-0.1015*x^4;
    ze_ = 5d0*tmax*p;
end


%Función que proporciona la coordenada z/c=zp_extray x/c=xp_extra del extradós en función de x/c
%en la cuerda geométrica
function zp_extra_ = zp_extra(x)
    ang = atan(zc_I(x));
    zp_extra_ = zcurv(x) + ze(x)*cos(ang);
end

function xp_extra_ = xp_extra(x)
    ang = atan(zc_I(x));
    xp_extra_ = x - ze(x)*sin(ang);
end


%Idem con intradós
function zp_intra_ = zp_intra(x)
   ang = atan(zc_I(x));
   zp_intra_ = zcurv(x) - ze(x)*cos(ang);
end

function xp_intra_ = xp_intra(x)
   ang = atan(zc_I(x));
   xp_intra_ = x + ze(x)*sin(ang);
end
    

