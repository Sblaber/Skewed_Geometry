#Load all the packages I might need
from __future__ import division # must be first
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.linalg import expm
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.special import erfi
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from datetime import datetime
startTime = datetime.now()
print(startTime)

## This code plots fig. 4 from https://doi.org/10.1063/5.0033405 ##
## It takes seconds to run ##

N2 = 15
dt_tau = np.zeros(N2)
Wex_F = np.zeros(N2)
Wex_R = np.zeros(N2)
dW_sqrdF = np.zeros(N2)
dW_cubeF = np.zeros(N2)
dW_quartF = np.zeros(N2)
dW_sqrdR = np.zeros(N2)
dW_cubeR = np.zeros(N2)
dW_quartR = np.zeros(N2)
Kappa4_F = np.zeros(N2)
Kappa4_R = np.zeros(N2)
Biaslin = np.zeros(N2)
Zeta3 = np.zeros(N2)
BiaslinR = np.zeros(N2)
VM = np.zeros(N2)
BNS = np.zeros(N2)
LR_lin = np.zeros(N2)
LR_linR = np.zeros(N2)
#zeta = np.zeros(N)
for i in range(0,N2):
    t_init = 0
    t_end  = 2.0**(i-5.0)
    N      = int(128*128*t_end) ### Compute N grid points
    dt     = float(t_end - t_init) / N
    ts    = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    k =  1
    kf = 0.5
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(ts[-1])
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    ks = 1.0*(k-c*ts)#1.0*k_appx_opt#1.0*(k-c*ts)
    aR = (k**(-0.5)-kf**(-0.5))/(ts[-1])
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    ksR = 1.0*(kf+c*ts)#1.0*k_appx_optR#1.0*(kf+c*ts)
    #ks[0] = 1.0*k
    #ks[-1] = 1.0*kf
    #ksR[-1] = 1.0*k
    #ksR[0] = 1.0*kf
    k_lin = k-c*ts
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    kdotlin = np.gradient(k-c*ts,ts)
    kdotappx = np.gradient(k_appx_opt,ts)
    k_bias_opt = 0.176009*(Deltat)**(3.0/2.0)/(2*ts+Deltat*0.314067)**(3.0/2.0)
    kdotbiasopt = np.gradient(k_bias_opt,ts)
    F = 0.5*np.log(kf/k)

    ##### Integration #####
    def fun1(t, y): return -2.0 * 1.0*ks[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun1, [ts[0], ts[-1]], [(1.0/ks[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun2(t, y): return -4.0 * 1.0*ks[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun2, [ts[0], ts[-1]], [(3.0/ks[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tF = 0.5*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_t,ts, initial=0)

    def fun3(t, y): return 2.0*W_tF[int(t/dt)] + 0.5*kdot[int(t/dt)]*xquart_t[int(t/dt)]  -2.0*ks[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun3, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun21(t, y): return kdot[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdFt = (solve_ivp(fun21, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    #W_sqrdFt = integrate.cumtrapz(np.gradient(ks,ts)*xsqrd_w_t,ts,initial=0)
    W_sqrdF = np.trapz(np.gradient(ks,ts)*xsqrd_w_t,ts)

    def fun4(t, y): return -6.0 * 1.0*ks[int(t/dt)] * y + 30.0 * xquart_t[int(t/dt)]
    x_hex_t = (solve_ivp(fun4, [ts[0], ts[-1]], [(15.0/(ks[0]**3.0))],method = 'Radau',t_eval=ts).y)[0]

    def fun5(t, y): return 12.0*xsqrd_w_t[int(t/dt)] + 0.5*kdot[int(t/dt)]*x_hex_t[int(t/dt)] - 4.0*ks[int(t/dt)]*y
    x_quart_w_t = (solve_ivp(fun5, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun6(t, y): return 2.0*W_sqrdFt[int(t/dt)] + kdot[int(t/dt)]*x_quart_w_t[int(t/dt)] - 2.0*ks[int(t/dt)]*y
    x_sqrd_wsqrd = (solve_ivp(fun6, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun22(t, y): return (3.0/2.0)*kdot[int(t/dt)]*x_sqrd_wsqrd[int(t/dt)]+0*y
    W_cubeFt = (solve_ivp(fun22, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]
    #W_cubeFt = (3.0/2.0)*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_wsqrd,ts,initial=0)
    W_cubeF = (3.0/2.0)*np.trapz(np.gradient(ks,ts)*x_sqrd_wsqrd,ts)

    def fun13(t, y): return -8.0 * 1.0*ks[int(t/dt)] * y + 56.0 * x_hex_t[int(t/dt)]
    x_oct_t = (solve_ivp(fun13, [ts[0], ts[-1]], [(105.0/(ks[0]**4.0))],method = 'Radau',t_eval=ts).y)[0]

    def fun14(t, y): return 30.0*x_quart_w_t[int(t/dt)] + 0.5*kdot[int(t/dt)]*x_oct_t[int(t/dt)] - 6.0*ks[int(t/dt)]*y
    x_hex_w_t = (solve_ivp(fun14, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun15(t, y): return 12.0*x_sqrd_wsqrd[int(t/dt)] + kdot[int(t/dt)]*x_hex_w_t[int(t/dt)] - 4.0*ks[int(t/dt)]*y
    x_quart_wsqrd = (solve_ivp(fun15, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun16(t, y): return 2.0*W_cubeFt[int(t/dt)] + (3.0/2.0)*kdot[int(t/dt)]*x_quart_wsqrd[int(t/dt)] - 2.0*ks[int(t/dt)]*y
    x_sqrd_wcube = (solve_ivp(fun16, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_quartF = (4.0/2.0)*np.trapz(np.gradient(ks,ts)*x_sqrd_wcube,ts)

    ##### Reverse #####
    def fun7(t, y): return -2.0 * 1.0*ksR[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun7, [ts[0], ts[-1]], [(1.0/ksR[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun8(t, y): return -4.0 * 1.0*ksR[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun8, [ts[0], ts[-1]], [(3.0/ksR[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tR = 0.5*integrate.cumtrapz(c*x_sqrd_t,ts, initial=0)

    def fun9(t, y): return 2.0*W_tR[int(t/dt)] + 0.5*kdotR[int(t/dt)]*xquart_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun9, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun23(t, y): return kdotR[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdRt = (solve_ivp(fun23, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    #W_sqrdRt = integrate.cumtrapz(c*xsqrd_w_t,ts,initial=0)
    W_sqrdR = np.trapz(c*xsqrd_w_t,ts)

    def fun10(t, y): return -6.0 * 1.0*ksR[int(t/dt)] * y + 30.0 * xquart_t[int(t/dt)]
    x_hex_t = (solve_ivp(fun10, [ts[0], ts[-1]], [(15.0/(ksR[0]**3.0))],method = 'Radau',t_eval=ts).y)[0]

    def fun11(t, y): return 12.0*xsqrd_w_t[int(t/dt)] + 0.5*kdotR[int(t/dt)]*x_hex_t[int(t/dt)] - 4.0*ksR[int(t/dt)]*y
    x_quart_w_t = (solve_ivp(fun11, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun12(t, y): return 2.0*W_sqrdRt[int(t/dt)] + kdotR[int(t/dt)]*x_quart_w_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    x_sqrd_wsqrd = (solve_ivp(fun12, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun24(t, y): return (3.0/2.0)*kdotR[int(t/dt)]*x_sqrd_wsqrd[int(t/dt)]+0*y
    W_cubeRt = (solve_ivp(fun24, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    #W_cubeRt = (3.0/2.0)*integrate.cumtrapz(c*x_sqrd_wsqrd,ts,initial=0)
    W_cubeR = (3.0/2.0)*np.trapz(c*x_sqrd_wsqrd,ts)

    def fun17(t, y): return -8.0 * 1.0*ksR[int(t/dt)] * y + 56.0 * x_hex_t[int(t/dt)]
    x_oct_t = (solve_ivp(fun17, [ts[0], ts[-1]], [(105.0/(ksR[0]**4.0))],method = 'Radau',t_eval=ts).y)[0]

    def fun18(t, y): return 30.0*x_quart_w_t[int(t/dt)] + 0.5*kdotR[int(t/dt)]*x_oct_t[int(t/dt)] - 6.0*ksR[int(t/dt)]*y
    x_hex_w_t = (solve_ivp(fun18, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun19(t, y): return 12.0*x_sqrd_wsqrd[int(t/dt)] + kdotR[int(t/dt)]*x_hex_w_t[int(t/dt)] - 4.0*ksR[int(t/dt)]*y
    x_quart_wsqrd = (solve_ivp(fun19, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun20(t, y): return 2.0*W_cubeRt[int(t/dt)] + (3.0/2.0)*kdotR[int(t/dt)]*x_quart_wsqrd[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    x_sqrd_wcube = (solve_ivp(fun20, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_quartR = (4.0/2.0)*np.trapz(c*x_sqrd_wcube,ts)

    tau = 1.0/(2.0*kf)
    dt_tau[i] = Deltat/tau
    Wex_F[i] = W_tF[-1]-F
    Wex_R[i] = W_tR[-1]+F
    dW_sqrdF[i] = W_sqrdF-2.0*F*W_tF[-1]+F**2.0#(W_sqrdF-W_tF[-1]**2)+(W_tF[-1]-F)**2.0
    dW_cubeF[i] =W_cubeF-3.0*F*W_sqrdF+3.0*F**2.0*W_tF[-1]-F**3.0#W_cubeF-3.0*W_tF[-1]*W_sqrdF+3.0*W_tF[-1]**2.0*W_tF[-1]-W_tF[-1]**3.0#W_cubeF-3.0*F*W_sqrdF+3.0*F**2.0*W_tF[-1]-F**3.0
    dW_quartF[i] = W_quartF-4*W_cubeF*F+6*W_sqrdF*F**2.0-4*W_tF[-1]*F**3.0+F**4.0#W_quartF-4*W_cubeF*W_tF[-1]+6*W_sqrdF*W_tF[-1]**2.0-4*W_tF[-1]*W_tF[-1]**3.0+W_tF[-1]**4.0#W_quartF-4*W_cubeF*F+6*W_sqrdF*F**2.0-4*W_tF[-1]*F**3.0+F**4.0
    #Kappa4_F[i] = dW_quartF[i]-3.0*(W_sqrdF-W_tF[-1]**2.0)**2.0
    dW_sqrdR[i] = W_sqrdR+2.0*F*W_tR[-1]+F**2.0#W_sqrdR-W_tR[-1]**2+(W_tR[-1]+F)**2.0
    dW_cubeR[i] = W_cubeR+3.0*F*W_sqrdR+3.0*F**2.0*W_tR[-1]+F**3.0#W_cubeR-3.0*W_tR[-1]*W_sqrdR+3.0*W_tR[-1]**2.0*W_tR[-1]-W_tR[-1]**3.0#W_cubeR+3.0*F*W_sqrdR+3.0*F**2.0*W_tR[-1]+F**3.0
    dW_quartR[i] = W_quartR+4*W_cubeR*F+6*W_sqrdR*F**2.0+4*W_tR[-1]*F**3.0+F**4.0#W_quartR-4*W_cubeR*W_tR[-1]+6*W_sqrdR*W_tR[-1]**2.0-4*W_tR[-1]*W_tR[-1]**3.0+W_tR[-1]**4.0#W_quartR+4*W_cubeR*F+6*W_sqrdR*F**2.0+4*W_tR[-1]*F**3.0+F**4.0
    #Kappa4_R[i] = dW_quartR[i]-3.0*(W_sqrdR-W_tR[-1]**2.0)**2.0
#### Bias Estimate ####
for i in range(0,N2):
    t_init = 0
    t_end  = 2.0**(i-5.0)
    N      = int(128*t_end) ### Compute 100 grid points
    zeta = np.zeros(N)
    ts = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    k =  1
    kf = 0.5
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(dt*N)
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    ks = np.float64(1.0*(k-c*ts))#1.0*k_appx_opt#1.0*(k-c*ts)
    aR = (k**(-0.5)-kf**(-0.5))/(dt*N)
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    ksR = np.float64(1.0*(kf+c*ts))#1.0*k_appx_optR#1.0*(kf+c*ts)
    #ks[0] = 1.0*k
    #ks[-1] = 1.0*kf
    #ksR[-1] = 1.0*k
    #ksR[0] = 1.0*kf
    k_lin = k-c*ts
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    kddot = np.gradient(kdot,ts)
    F = 0.5*np.log(kf/k)
    Biaslin[i] = -(1.0/2.0)*np.trapz(kdot**3/ks**5,ts)
    LR_lin[i] = (1.0/4.0)*np.trapz(kdot**2/ks**3,ts)#(1.0/4.0)*np.trapz(kdot*zeta,ts)#(1.0/4.0)*np.trapz(kdot**2/ks**3,ts)#(1.0/4.0)*np.trapz(kdot*zeta,ts)#
    LR_linR[i] = (1.0/4.0)*np.trapz(kdot**2/ks**3,ts)
    BiaslinR[i] = -(1.0/4.0)*np.trapz((kdotR**3/ksR**5),ts)
    #x_x_x_x = np.zeros((N+1,N+1,N+1,N+1))
    #for j in range(0,N+1):
    #    for l in range(0,N+1):
    #        for m in range(0,N+1):
    #            x_x_x_x[:,j,l,m] = 6.0*(60.0/ks**4.0)*np.exp(-0.5*ks*(ts[j]+ts[l]+ts[m]))*np.exp(-0.5*ks*np.abs(ts[j]-ts[l]-ts[m]))*np.exp(-0.5*ks*np.abs(ts[l]-ts[j]-ts[m]))*np.exp(-0.5*ks*np.abs(ts[m]-ts[l]-ts[j]))
    #Zeta3[i] = (1.0/16.0)*np.trapz(kdot**4.0*np.trapz(np.trapz(np.trapz(x_x_x_x,ts),ts),ts),ts)#(45.0/16.0)*np.trapz(kdot**4.0/ks**7.0,ts)
    #Cokurt = (np.exp(-6*ks*ts)*(np.exp(2*ks*ts)*(np.exp(2*ks*ts)*((12*ks*ts*(2*ks*ts-7)+125)*np.exp(2*ks*ts)-276)+219)-68))/(48*ks**7)
    Zeta3[i] = (27.0/16.0)*np.trapz(kdot**4.0/ks**7.0,ts)+(12.0/16.0)*np.trapz(kdot**4.0*(ts/(4*ks**6.0)-1.0/(8*ks**7.0)),ts)
    #plt.plot(ts/t_end,-(1.0/4.0)*c/k_lin**3,'--')

for i in range(0,1):
    #t_end = 10*2**i
    #Nvec = np.array(np.loadtxt('Nvec_t_end_{}.txt'.format(t_end)))
    #S_F = np.array(np.loadtxt('S_F_t_end_{}.txt'.format(t_end)))
    #S_R = np.array(np.loadtxt('S_R_t_end_{}.txt'.format(t_end)))
    #ES_F = np.array(np.loadtxt('ES_F_t_end_{}.txt'.format(t_end)))
    #ES_R = np.array(np.loadtxt('ES_R_t_end_{}.txt'.format(t_end)))
    #Kurt_F = np.array(np.loadtxt('Kurt_F_t_end_{}.txt'.format(t_end)))
    #Kurt_R = np.array(np.loadtxt('Kurt_R_t_end_{}.txt'.format(t_end)))
    #EKurt_F = np.array(np.loadtxt('EKurt_F_t_end_{}.txt'.format(t_end)))
    #EKurt_R = np.array(np.loadtxt('EKurt_R_t_end_{}.txt'.format(t_end)))
    plt.figure(1)
    plt.plot(dt_tau**(1.0),LR_lin,'k',label = 'Approximation')
    plt.loglog(dt_tau**(1.0),0.5*(Wex_F+Wex_R),'k--',label = '$n=1$')
    plt.loglog(dt_tau**(1.0),0.25*(dW_sqrdF+dW_sqrdR),'k:',label = '$n=2$', linewidth = 2.0)
    plt.loglog(dt_tau**(1.0),0.25*(dW_cubeR+dW_cubeF),'k-.',label = '$n=3$') #1/4 vs 1/2???
    plt.legend(frameon=False,fontsize = 12)
    plt.plot(dt_tau**(1.0),LR_lin,'k',label = '(+)')
    plt.plot(dt_tau**(1.0),Biaslin,'r',label = '(-)')
    plt.loglog(dt_tau**(1.0),2.0*np.abs(Wex_F-Wex_R),'r--',label = '(-)')
    plt.loglog(dt_tau**(1.0),(1.0/3.0)*np.abs(dW_sqrdF-dW_sqrdR),'r:', linewidth = 2.0)
    plt.loglog(dt_tau**(1.0),(1.0/6.0)*(dW_cubeR-dW_cubeF),'r-.') #1/6 vs 1/4?
    plt.plot(dt_tau**(1.0),(1.0)*Zeta3,'k')
    plt.xlim(10**(-1.0),10**(2.0))
    plt.ylim(10**(-5.0),10**(0.0))
    plt.xticks([0.1,1,10,100],fontsize = 12)
    plt.yticks([0.0001,0.01,1],fontsize = 12)
    plt.ylabel('$\\alpha^{\\pm}_{n}\\beta^{n}\\left(\\langle W_{\\rm ex}^{n} \\rangle_{\Lambda}\\pm\\langle W_{\\rm ex}^{n} \\rangle_{\Lambda^{\dagger}}\\right) $',fontsize = 16)
    plt.xlabel('$\Delta t/\\tau^{(1)}_{\\rm f}$',fontsize = 16)
    #plt.grid()
    plt.tight_layout()
    plt.savefig("Work_Var_skew_1.pdf")

    plt.figure(2)
    plt.loglog(dt_tau**(-1.0),(dW_cubeR-dW_cubeF),'r')
    plt.plot(dt_tau**(-1.0),6.0*Biaslin,'r--')
    #plt.errorbar(0*S_F+t_end**(-1.0),np.abs(S_F+S_R),yerr=np.abs(S_F+S_R)*np.sqrt((ES_F/S_F)**2.0+(ES_R/S_R)**2.0),fmt = 'ro')
    plt.plot(dt_tau**(-1.0),(dW_cubeF+dW_cubeR),'k')
    plt.plot(dt_tau**(-1.0),(4.0)*Zeta3,'k--')
    #plt.plot(dt_tau**(-1.0),(12.0)*LR_lin**2.0,'k--')
    #plt.errorbar(0*S_F+t_end**(-1.0),np.abs(S_F-S_R),yerr=np.abs(S_F-S_R)*np.sqrt((ES_F/S_F)**2.0+(ES_R/S_R)**2.0),fmt = 'ko')
    #plt.errorbar(0*Kurt_F+t_end**(-1.0),0.5*np.abs(Kurt_F+Kurt_R),yerr=np.abs(Kurt_F+Kurt_R)*np.sqrt((EKurt_F/Kurt_F)**2.0+(EKurt_R/Kurt_R)**2.0),fmt = 'go')
    plt.loglog(dt_tau**(-1.0),0.5*(dW_quartF+dW_quartR),'g')
    plt.xlim(10**(-2.0),10**(1.0))
    plt.ylim(10**(-5.0),10**(4.0))
    plt.ylabel('$\\langle  W_{\\rm ex}^{3} \\rangle_{\Lambda}\\pm\\langle  W_{\\rm ex}^{3} \\rangle_{\Lambda^{\dagger}} $',fontsize = 16)
    plt.xlabel('$\\tau/\Delta t$',fontsize = 16)
    plt.grid()
    plt.tight_layout()

    plt.figure(3)
    plt.loglog(dt_tau**(-1.0),(dW_quartF+dW_quartR),'k')
    #plt.loglog(dt_tau**(-1.0),2.0*(dW_quartF),'g')
    #plt.loglog(dt_tau**(-1.0),2.0*(dW_quartR),'b')
    #plt.errorbar(0*Kurt_F+t_end**(-1.0),np.abs(Kurt_F+Kurt_R),yerr=np.abs(Kurt_F+Kurt_R)*np.sqrt((EKurt_F/Kurt_F)**2.0+(EKurt_R/Kurt_R)**2.0),fmt = 'ko')
    plt.plot(dt_tau**(-1.0),(24.0)*LR_lin**2.0,'k--')
    plt.loglog(dt_tau**(-1.0),np.abs(dW_quartF-dW_quartR),'r')
    plt.plot(dt_tau**(-1.0),(20.0)*LR_lin**1.0*3.0*Biaslin,'r--')
    plt.xlim(10**(-2.0),10**(1.0))
    plt.ylim(10**(-5.0),10**(4.0))
    plt.ylabel('$\\langle  W_{\\rm ex}^{4} \\rangle_{\Lambda}\\pm\\langle  W_{\\rm ex}^{4} \\rangle_{\Lambda^{\dagger}} $',fontsize = 16)
    plt.xlabel('$\\tau/\Delta t$',fontsize = 16)
    plt.grid()
    plt.tight_layout()

    #plt.figure()
    #plt.errorbar(Nvec,np.abs(S_F+S_R),yerr=np.abs(S_F+S_R)*np.sqrt((ES_F/S_F)**2.0+(ES_R/S_R)**2.0),fmt = 'ro')
    #plt.errorbar(Nvec,np.abs(S_F-S_R),yerr=np.abs(S_F-S_R)*np.sqrt((ES_F/S_F)**2.0+(ES_R/S_R)**2.0),fmt = 'ko')
plt.show()
print(datetime.now()-startTime)
