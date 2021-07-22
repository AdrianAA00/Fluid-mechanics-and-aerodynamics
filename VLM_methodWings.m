%% Authors:
% Antón , Adrián
% Tabanera , Antonio
% Balbuena , Daniel
% Larrubia , Raúl

%% METODO DE MALLAS DE TORBELLINOS PARA ALAS DA GRAN ENVERGADURA (alargamiento>5, estrechamiento>4)
%%% ala trapezoidal
%%% vuelo simétrico
clear ; clc; close all

%% CONDICIONES DE VUELO

Vinf=1.0; %velocidad de la corriente incidente

%% PARAMETROS ALA TRAPEZOIDAL

%%Elección perfil NACAXYZK
global xfmax fmax tmax;     %parámetros para el perfil NACA de 4 cifras
%Cifras XYZK = 8884,5 
fmax = (1d0 + fix(8/2d0))/100d0;%4/100;%
xfmax =(20d0 + 10d0*fix(8d0/4d0))/100d0; %40/100;%
tmax =(7d0 + fix(84.5/9d0))/100d0;% 12/100;%


%%% Parámetros del ala
global lambda delta A Sw Ny b torsion; 

Ny = 100;              %Número paneles en el eje Y por ala

lambda = 0.4;        %Estrechamiento
delta = 15;            %Flecha (angulo) en grados de la cuerda 1/4
A = 8;                %Alargamiento
torsion=-2;         %Torsion en la punta del ala en grados (su distribución se define como una funcion)
Sw = 1;               %Superficie total de las 2 alas (dcha e izq)

%%% Cálculos para el ala derecha (y positivo)
b = sqrt(Sw*A);        %Envergadura; S = b^2/ A
Cgm = b/A      ;       %Cuerda media
C0 = 2*Cgm/(1+lambda); %Cuerda en la raiz
Ct = C0*lambda ;       %Cuerda en el borde marginal
Cam =  (2/3)*C0*(1+lambda+lambda^2)/(1+lambda) ; %Cuerda aerodinamica media

d = (b/2)*tan(deg2rad(delta))-Ct/4+C0/4;
delta_BA = rad2deg(atan(2*d/b))      ;               %angulo de la linea del borde de ataque
delta_1_4 =  delta;                                  %angulo de la linea de 1/4 de la cuerda
delta_1_2 = rad2deg(atan((d+2*Ct/4-2*C0/4)/(b/2)));  %angulo de la linea de 1/2 de la cuerda
delta_3_4 = rad2deg(atan((d+3*Ct/4-3*C0/4)/(b/2))) ; %angulo de la linea de 3/4 de la cuerda
delta_BS = rad2deg(atan((d+4*Ct/4-4*C0/4)/(b/2)))   ;%angulo de la linea del borde de salida

%Puntos de control en la linea 3/4
y_c(1,Ny)=0.0; x_c(1,Ny)=0.0;
for i = 1:Ny
  y_c(i) = - b/(4*Ny) + (b*i)/(2*Ny);
  x_c(i) = y_c(i)*tan(deg2rad(delta_3_4)) + 3*C0/4; 
end

%Coordenadas en la linea 1/4 en el centro de cada panel
y_c_14(1,Ny)=0.0; x_c_14(1,Ny)=0.0;
for i = 1:Ny
  y_c_14(i) = - b/(4*Ny) + (b*i)/(2*Ny);
  x_c_14(i) = y_c_14(i)*tan(deg2rad(delta)) + 1*C0/4; 
end

%4 Esquinas de cada panel          Numeración  1 - 2    + - - >y
                                           %   |   |    |
                                           %   4 - 3    x
y_panel(Ny,4)=0; x_panel(Ny,4)=0; 
for i = 1:Ny                               
  y_panel(i,1) = (b*(i-1))/(2*Ny);         
  y_panel(i,2) = (b*(i))/(2*Ny);
  y_panel(i,3) = (b*(i))/(2*Ny);
  y_panel(i,4) = (b*(i-1))/(2*Ny);
  x_panel(i,1) =  y_panel(i,1)*tan(deg2rad(delta_BA)); 
  x_panel(i,2) =  y_panel(i,2)*tan(deg2rad(delta_BA));   
  x_panel(i,3) =  y_panel(i,3)*tan(deg2rad(delta_BS)) + C0;
  x_panel(i,4) =  y_panel(i,4)*tan(deg2rad(delta_BS)) + C0; 
end


%Cuerda media y envergadura media de cada panel
b_media(1,Ny)=0.0; c_media(1,Ny)=0.0;
for i = 1:Ny
  b_media(i) = b/(2*Ny);  
  c_media(i) = C0 + y_c(i)*(tan(deg2rad(delta_BS))-tan(deg2rad(delta_BA)));
end

%Extremos de cada panel en linea 1/4
y1(1,Ny)=0.0;y2(1,Ny)=0.0;x1(1,Ny)=0.0;x2(1,Ny)=0.0;
for i = 1:Ny  
  y1(i) = (b*(i-1))/(2*Ny);
  y2(i) = (b*(i))/(2*Ny);
  
  x1(i) =  y1(i)*tan(deg2rad(delta)) + 1*C0/4;
  x2(i) =  y2(i)*tan(deg2rad(delta)) + 1*C0/4;  
end

%Centro aerodinámico teórico
Xca = 0.25*C0 + tan(deg2rad(delta))*(b/2)*(1+2*lambda)/(3*(1+lambda));
Yca = (b/2)*(1+2*lambda)/(3*(1+lambda));

%Cálculo pendiente del perfil que vamos a usar en 3/4
dy_dx =zc_I(3/4);

%% BUCLE PARA CALCULAR LOS COEFICIENTES PARA DISTINTOS ÁNGULOS DE ATAQUE

for p=1:451
    alpha(p)=(-10 + 20/400*(p-1))*(2*pi/360);                     %Para obtener mayor precision con alpha para sustentacion nula (no afecta al resto de representaciones)
    alpha_grados(p)=alpha(p)*360/(2*pi);                                 %%%vector de alfas 

V_lig_d(1,Ny)=0.0;V_lib_d(1,Ny)=0.0;V_lig_i(1,Ny)=0.0;V_lib_i(1,Ny)=0.0;V_lig_i(1,Ny)=0.0;V_i(1,Ny)=0.0;V_d(1,Ny)=0.0;A_zj(Ny,Ny)=0.0;B_j(1,Ny)=0;
%am(Ny,Ny)=0;bm(Ny,Ny)=0;cm(Ny,Ny)=0;dm(Ny,Ny)=0;em(Ny,Ny)=0;fm(Ny,Ny)=0;gm(Ny,Ny)=0;hm(Ny,Ny)=0;im(Ny,Ny)=0;jm(Ny,Ny)=0;km(Ny,Ny)=0;lm(Ny,Ny)=0;
%ai(Ny,Ny)=0;bm(Ny,Ny)=0;cm(Ny,Ny)=0;dm(Ny,Ny)=0;em(Ny,Ny)=0;fm(Ny,Ny)=0;gm(Ny,Ny)=0;hm(Ny,Ny)=0;im(Ny,Ny)=0;jm(Ny,Ny)=0;km(Ny,Ny)=0;lm(Ny,Ny)=0;
for j=1:Ny
    for z=1:Ny
        %%% velocidad inducida por un panel inductor z sobre un punto de control j por un torbellino de valor unidad
        %%% torellino inductor del semiala derecha
        a=x_c(j)-x1(z); %am(j,z)=a;
        b_=y_c(j)-y1(z);%bm(j,z)=b_;
        c=x_c(j)-x2(z);%cm(j,z)=c;
        d=y_c(j)-y2(z);%dm(j,z)=d;
        e=sqrt(a^2+b_^2);%em(j,z)=e;
        f=sqrt(c^2+d^2);%fm(j,z)=f;
        g=x2(z)-x1(z);%gm(j,z)=g;
        h=y2(z)-y1(z);%hm(j,z)=h;
        k=(g*a+h*b_)/e-(g*c+h*d)/f;%km(j,z)=k;
        l=-(1/b_)*(1+a/e)+1/d*(1+c/f);%lm(j,z)=l;
        V_lig_d(z)=k/(a*d-b_*c);                     %velocidad inducida en el panel j por el torbellino ligado z de la derecha
        V_lib_d(z)=l;                               %velocidad inducida en el panel j por el torbellino libre z de la derecha
        V_d(j,z)=1/(4*pi)*(V_lig_d(z)+V_lib_d(z));    %velocidad inducida en el panel j por el torbellino z de la derecha
        
        %%% torbellino inductor del semiala izquierda. Se hace de esta forma por la simetria del problema
        a=x_c(j)-x2(z);
        b_=y_c(j)+y2(z);
        c=x_c(j)-x1(z);
        d=y_c(j)+y1(z);
        e=sqrt(a^2+b_^2);
        f=sqrt(c^2+d^2);
        g=x1(z)-x2(z);
        h=-y1(z)+y2(z);
        k=(g*a+h*b_)/e-(g*c+h*d)/f;
        l=-(1/b_)*(1+a/e)+1/d*(1+c/f);
        V_lig_i(z)=k/(a*d-b_*c);                 %velocidad inducida en el panel j por el torbellino ligado z de la izquierda
        V_lib_i(z)=l;                           %velocidad inducida en el panel j por el torbellino libre z de la izquierda
        V_i(j,z)=1/(4*pi)*(V_lig_i(z)+V_lib_i(z));    %velocidad inducida en el panel j por el torbellino z de la izquierda
      
        A_zj(z,j)=V_i(j,z)+V_d(j,z);    
    end
    

    B_j(j)=alpha(p)+epsilon(y_c(j))-dy_dx;
  
end

gamma=-A_zj^(-1)*((B_j)');      %Distribucion de torbellinos para una velocidad infinito unitaria

%% CARACTERISTICAS AERODINAMICAS

Cl(1,Ny)=0.0;Cm0Y(1,Ny)=0.0;alpha_i(1,Ny)=0.0;Cdi(1,Ny)=0.0;CL=0.0;CM0Y=0.0;CDi=0.0;
for j=1:Ny
    Cl(j)=2*gamma(j)/c_media(j);                        %%%coeficiente de sustentacion de cada panel
    CL=CL+(2/Sw)*(Cl(j)*b_media(j)*c_media(j));         %%%Coeficiente de sustentacion global del ala
   
    Cm0Y(j)=-Cl(j)*x_c_14(j)/C0;                        %%%coeficiente de momentos de cada panel respecto del eje Y
    CM0Y=CM0Y+(2/Sw)*(Cm0Y(j)*b_media(j)*c_media(j));   %%%coeficiente de momentos global del ala respecto del eje Y
    
    alpha_i(j)= gamma(j)*(V_lib_d(j)+V_lib_i(j))/Vinf;  %%%angulo de ataque inducido
    Cdi(j)=Cl(j)*alpha_i(j);                            %%%coeficiente de resistencia inducida en cada panel
    CDi=CDi+(2/Sw)*(Cdi(j)*b_media(j)*c_media(j));      %%%coeficiente de resistencia inducida total en el ala
end
    gamma_alpha(:,p)=gamma(:);                             %%%distribucion de torbellinos
    Cl_alpha(:,p)=Cl;                                      %%%distribucion de coeficiente de sustentacion en cada punto del ala derecha en funcion de alpha
    CL_alpha(p)=CL;                                        %%%coeficiente de sustentacion global en funcion de alfa
    Cdi_alpha(:,p)=Cdi;                                    %%%distribucion de coeficiente de resistencia inducido en cada punto del ala derecha en funcion de alpha
    CDi_alpha(p)=CDi;                                      %%%coeficiente de sustentacion global en funcion de alfa
    CM0Y_alpha(p)=CM0Y;                                    %%%coeficiente de momentos respecto de el eje Y
    CMca(p)=CM0Y+CL*Xca/C0;                                %%%coeficiente de momentos respecto en el centro aerodinamico en un eje paralelo al eje Y 
    
    X_A=0.7;                                               %%%posicion de un eje arbitrario (paralelo al eje Y) respecto del que calcular el coeficiente de momentos
    CM_A(p)=CM0Y+CL*X_A/C0;                                %%%coeficiente de momentos respecto de un eje arbitrario en X_A
    
end


%% Determinar el centro aerodinámico real por prueba y error, no por la posición teorica aproximada
     incremento=-0.0046322;         %%%Lo que te alejas en X del centro aerodinamico teorico
     X_A=Xca+incremento;                                             %%%posicion de un eje arbitrario (paralelo al eje Y) respecto del que calcular el coeficiente de momentos
     CM_A=CM0Y_alpha+CL_alpha*X_A/C0; 
     error=max(CM_A)-min(CM_A);       %%%Debería ser nulo el error, así el CM no varía con el angulo de ataque  
    
%% Gráfica de distribución de torbellinos
indices(1,5)=0.0;
indices=[find(alpha_grados==-10),find(alpha_grados==-5),find(alpha_grados==0),find(alpha_grados==5),find(alpha_grados==10)];
alfa_indices=[alpha_grados(indices(1)),alpha_grados(indices(2)),alpha_grados(indices(3)),alpha_grados(indices(4)),alpha_grados(indices(5))]; %Se determinan los indices en los que se dan los angulos de aqtaque que se quieren

%%%distribucion de torbellinos
for i=1:5
figure(i+32)
scatter(y_c,gamma_alpha(:,indices(i)),'b');
hold on
scatter(-y_c,gamma_alpha(:,indices(i)),'b');
hold on
plot(-y_c,gamma_alpha(:,indices(i)),'r',y_c,gamma_alpha(:,indices(i)),'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('distribucion de torbellinos','Fontsize', 20)
legend('gamma')
title('Distribucion de torbellinos vs yc para alfa','Fontsize', 15)
ymin=min(gamma_alpha(:,indices(i)))-0.3;
ymax=max(gamma_alpha(:,indices(i)))+0.3;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
end


%% Gráfica de distribución de Cl y de Cd_inducido para ángulos de -10º, -5º, 0º,5º,10º
indices(1,5)=0.0;
indices=[find(alpha_grados==-10),find(alpha_grados==-5),find(alpha_grados==0),find(alpha_grados==5),find(alpha_grados==10)];
alfa_indices=[alpha_grados(indices(1)),alpha_grados(indices(2)),alpha_grados(indices(3)),alpha_grados(indices(4)),alpha_grados(indices(5))]; %Se determinan los indices en los que se dan los angulos de aqtaque que se quieren


%%% AMBOS ALAS
%%%Cl
for i=1:5
figure(i)
scatter(y_c,Cl_alpha(:,indices(i)),'b');
hold on
scatter(-y_c,Cl_alpha(:,indices(i)),'b');
hold on
plot(-y_c,Cl_alpha(:,indices(i)),'r',y_c,Cl_alpha(:,indices(i)),'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl','Fontsize', 20)
legend('Cl')
title('Distribucion de Cl vs yc para alfa =','Fontsize', 15)
ymin=min(Cl_alpha(:,indices(i)))-0.5;
ymax=max(Cl_alpha(:,indices(i)))+0.5;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
end

%%%Cdi
for i=1:5
figure(i+5)
scatter(y_c,Cdi_alpha(:,indices(i)),'b');
hold on
scatter(-y_c,Cdi_alpha(:,indices(i)),'b');
hold on
plot(-y_c,Cdi_alpha(:,indices(i)),'r',y_c,Cdi_alpha(:,indices(i)),'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cd_{inducido}','Fontsize', 20)
legend('Cd_{inducido}')
title('Distribucion de Cd_{inducido} vs yc para alfa =','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
end



%%% SOLO ALA DERECHA
%%%Cl
for i=1:5
figure(i+10)
scatter(y_c,Cl_alpha(:,indices(i)));
alpha_grados(indices(i))
CL_alpha(indices(i))
CDi_alpha(indices(i))
hold on
plot(y_c,Cl_alpha(:,indices(i)),'LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl','Fontsize', 20)
legend('Cl')
title('Distribucion de Cl vs yc para alfa =','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
end

%%%Cdi
for i=1:5
figure(i+15)
scatter(y_c,Cdi_alpha(:,indices(i)));
hold on
plot(y_c,Cdi_alpha(:,indices(i)),'LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cd_{inducido}','Fontsize', 20)
legend('Cd_{inducido}')
title('Distribucion de Cd_{inducido} vs yc para alfa =','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
end

%% GRAFICAS EN FUNCION DE ALPHA: SUSTENTACION, RESISTENCIA INDUCIDA Y MOMENTOS

%%%Pendiente de la recta CL-alpha
pendiente_CL_alfha=(CL_alpha(80)-CL_alpha(20))/(alpha(80)-alpha(20));
pendiente_CL_alfha_grados=(CL_alpha(80)-CL_alpha(20))/(alpha_grados(80)-alpha_grados(20));

%%%GRÁFICA CL en funcion de alfa 

figure(21)
plot(alpha_grados,CL_alpha,'LineWidth',2);
xlabel('alfa','Fontsize', 20)
ylabel('CL','Fontsize', 20)
legend('CL vs α')
title('CL en funcion de alfa','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;
 
%%%GRÁFICA CD_inducido en funcion de alfa

figure(22)
plot(alpha_grados,CDi_alpha,'LineWidth',2);
xlabel('alfa','Fontsize', 20)
ylabel('CD_{inducido}','Fontsize', 20)
legend('CD_{inducido} vs α')
title('CD_{inducido} en funcion de alfa','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%%%CURVA POLAR del perfil: CD vs CL

figure(23)
plot(CL_alpha,CDi_alpha,'LineWidth',2);
xlabel('CL','Fontsize', 20)
ylabel('CD_{inducido}','Fontsize', 20)
legend('CD_{inducido} vs CL')
title('POLAR DEL ALA','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%%%GRAFICAR coeficiente de momentos en funcion del angulo de ataque

figure(24)
plot(alpha_grados,CMca,'r',alpha_grados,CM_A,'b',alpha_grados,CM0Y_alpha,'g','LineWidth',2);
title('Coeficientes de momentos')
ylabel('Coeficientes de momentos')
xlabel('alpha')
legend('CMca [teorico]','CM [centro aearodinamico real]','CM0Y')

%% Caso de alpha=0 y distribucion de sustentacion inicial

%%%Cálculo del ángulo nulo
[alpha_nulo,indice_CL_alpha_nulo]=min(abs(alpha));
CL_alpha_nulo=CL_alpha(indice_CL_alpha_nulo);
 
%%%Distribución de sustentación inicial
Cl_inicial=Cl_alpha(:,indice_CL_alpha_nulo)';

%%%comprobación del CL inicial
CL_inicial=0.0;
for i=1:Ny 
    CL_inicial=CL_inicial+(2/Sw)*(Cl_inicial(i)*b_media(i)*c_media(i));
end 

%% Distribución de sustentación básica (por definicion) --> genera CL=0

%%%Cálculo del ángulo y del indice donde se consigue CL=0
[CL_nulo,indice_CL_nulo]=min(abs(CL_alpha));
alpha_CL_nulo=alpha(indice_CL_nulo);
alpha_CL_nulo_grados=alpha_grados(indice_CL_nulo);

%%%Distribución de sustentación básica
Cl_basico2=Cl_alpha(:,indice_CL_nulo)';

%%%comprobación del CL basico--> debe ser cero
CL_basico2=0.0;
for i=1:Ny 
    CL_basico2=CL_basico2+(2/Sw)*(Cl_basico2(i)*b_media(i)*c_media(i));
end 

%GRAFICA distribución de sustentación básica
figure(25)
scatter(y_c,Cl_basico2);
hold on
plot(y_c,Cl_basico2,'LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{basico}','Fontsize', 20)
legend('','Cl_{basico}')
title('Distribucion de Cl vs yc con sustentacion total nula, alpha','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

figure(26)
scatter(y_c,Cl_basico2,'b');
hold on
scatter(-y_c,Cl_basico2,'b');
hold on
plot(-y_c,Cl_basico2,'r',y_c,Cl_basico2,'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{basico}','Fontsize', 20)
legend('','','Cl_{basico}','')
title('Distribucion de Cl vs yc con sustentacion total nula, alpha','Fontsize', 15)
ymin=min(Cl_alpha(:,indice_CL_nulo))-0.5;
ymax=max(Cl_alpha(:,indice_CL_nulo))+0.5;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%% Distribución de Cl adicional y basico con CL=1: Cl=Cl_basico+Cl_adicional*CL

%%%Cálculo del ángulo y del indice donde se consigue CL=1
[cero_teorico,indice_CL_1]=min(abs(CL_alpha-1));      %el primer termino debería ser cero teoricamente para tener el valor exacto
CL_1=CL_alpha(indice_CL_1);
alpha_CL_1=alpha(indice_CL_1);
alpha_CL_1_grados=alpha_CL_1*360/(2*pi);     %Valor de alpha en grados

%%%resolviendo el sistema de ecuaciones se obtienen Cl_basico y Cl_adicional
Cl_adicional(:)=(Cl_alpha(:,(p-1)/2)-Cl_alpha(:,(p-1)/6))/(CL_alpha((p-1)/2)-CL_alpha((p-1)/6));
Cl_basico(:)=Cl_alpha(:,(p-1)/2)-Cl_adicional(:)*CL_alpha((p-1)/2);

%%%CL por la distribucion adicional
CL_adicional=0.0;
for i=1:Ny 
    CL_adicional=CL_adicional+(2/Sw)*(Cl_adicional(i)*b_media(i)*c_media(i));
end 

%%%CL por la distribucion basica
CL_basico=0.0;
for i=1:Ny 
    CL_basico=CL_basico+(2/Sw)*(Cl_basico(i)*b_media(i)*c_media(i));
end 

%GRAFICA distribución de sustentación básica ala derecha
figure(27)
scatter(y_c,Cl_basico);
hold on
plot(y_c,Cl_basico,'LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{basico}','Fontsize', 20)
legend('','Cl_{basico}')
title('Distribucion de Cl_{basico}','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%GRAFICA distribución de sustentación básica en los dos alas
figure(28)
scatter(y_c,Cl_basico,'b');
hold on
scatter(-y_c,Cl_basico,'b');
hold on
plot(-y_c,Cl_basico,'r',y_c,Cl_basico,'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{basico}','Fontsize', 20)
legend('','','Cl_{basico}','')
title('Distribucion de Cl_{basico}','Fontsize', 15)
ymin=min(Cl_basico(:))-0.5;
ymax=max(Cl_basico(:))+0.5;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%GRAFICA distribución de sustentación adicional ala derecha
figure(29)
scatter(y_c,Cl_adicional);
hold on
plot(y_c,Cl_adicional,'LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{adicional}','Fontsize', 20)
legend('','Cl_{adicional}')
title('Distribucion de Cl_{adicional}','Fontsize', 15)
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%GRAFICA distribución de sustentación adicional ala derecha
figure(30)
scatter(y_c,Cl_adicional,'b');
hold on
scatter(-y_c,Cl_adicional,'b');
hold on
plot(-y_c,Cl_adicional,'r',y_c,Cl_adicional,'r','LineWidth',2);
hold off
xlabel('y control','Fontsize', 20)
ylabel('Cl_{adicional}','Fontsize', 20)
legend('','','Cl_{adicional}')
title('Distribucion de Cl_{adicional}','Fontsize', 15)
ymin=min(Cl_adicional)-0.5;
ymax=max(Cl_adicional)+0.5;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;


%%%TODAS LAS GRAFICAS JUNTAS:
%%%Grafica de las distribuciones adicional y unitaria para el ala derecha

figure(31)
plot(y_c,Cl_alpha(:,indice_CL_1),'b','LineWidth',2);
hold on 
scatter(y_c,Cl_alpha(:,indice_CL_1),'b');
hold on 
plot(y_c,Cl_basico,'r','LineWidth',2);
hold on 
scatter(y_c,Cl_basico,'r');
hold on 
plot(y_c,Cl_adicional,'g','LineWidth',2);
hold on 
scatter(y_c,Cl_adicional,'g');
hold off 

xlabel('y control','Fontsize', 5)
ylabel('Cl','Fontsize', 5)
legend('Cl_{CL=1}', '','Cl basico', '', 'Cl adicional', '') 
title('Sustentacion basica y adicional para CL=1');
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%%%Grafica de las distribuciones adicional y unitaria para los dos alas
figure(32)
plot(-y_c,Cl_alpha(:,indice_CL_1),'b',y_c,Cl_alpha(:,indice_CL_1),'b','LineWidth',2);
hold on 
scatter(-y_c,Cl_alpha(:,indice_CL_1),'b');
hold on 
scatter(y_c,Cl_alpha(:,indice_CL_1),'b');
hold on 
plot(-y_c,Cl_basico,'r',y_c,Cl_basico,'r','LineWidth',2);
hold on 
scatter(-y_c,Cl_basico,'r');
hold on 
scatter(y_c,Cl_basico,'r');
hold on 
plot(y_c,Cl_adicional,'g',y_c,Cl_adicional,'g','LineWidth',2);
hold on 
scatter(y_c,Cl_adicional,'g');
hold on 
scatter(-y_c,Cl_adicional,'g');
hold off 

xlabel('y control','Fontsize', 5)
ylabel('Cl','Fontsize', 5)
legend('Cl_{CL=1}','','','','Cl basico','', '','', 'Cl adicional','', '','') 
title('Sustentacion basica y adicional para CL=1');
ymin=min(Cl_basico)-0.7;
ymax=max(Cl_alpha(:,indice_CL_1))+0.7;
axis([-1.5 1.5 ymin ymax])
ax = gca;
ax.XAxis.FontSize = 10;
ax.YAxis.FontSize = 10;

%% FUNCIONES

%Cálculo de la derivada analítica de la cuerda perfil en linea 3/4
function zc_I_ = zc_I(x)
    global xfmax fmax
    
    if (x >= 0d0) && (x <= xfmax)
        zc_I_ = (fmax/xfmax^2)*(2d0*xfmax-2d0*x);
    elseif (x <= 1.0001) && (x >= xfmax)
        zc_I_ = (fmax/(1-xfmax)^2)*(2d0*xfmax-2d0*x);
    end
end


function torsion_=epsilon(y)
    global b torsion
    
    torsion_=torsion*y/(b/2)*(2*pi/360);
    
end
