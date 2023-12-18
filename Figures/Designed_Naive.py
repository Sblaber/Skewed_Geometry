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

## This code plots fig. 2 from https://doi.org/10.1063/5.0033405 ##
## It takes ~10 minutes to run ##

# For high accuracy calculation I set N = int(4000*t_end) in every loop.

N2 = 53
dt_tau = np.zeros(N2)
Wex_F_lin = np.zeros(N2)
Wex_R_lin = np.zeros(N2)
dW_sqrdF_lin = np.zeros(N2)
dW_sqrdR_lin = np.zeros(N2)
Wex_F_LR = np.zeros(N2)
Wex_R_LR = np.zeros(N2)
dW_sqrdF_LR = np.zeros(N2)
dW_sqrdR_LR = np.zeros(N2)
Wex_F_Biasopt = np.zeros(N2)
Wex_R_Biasopt = np.zeros(N2)
dW_sqrdF_Biasopt = np.zeros(N2)
dW_sqrdR_Biasopt = np.zeros(N2)
Wex_F_FI = np.zeros(N2)
Wex_R_FI = np.zeros(N2)
dW_sqrdF_FI = np.zeros(N2)
dW_sqrdR_FI = np.zeros(N2)
Wex_F = np.zeros(N2)
Wex_R = np.zeros(N2)
dW_sqrdF = np.zeros(N2)
dW_sqrdR = np.zeros(N2)
Biaslin = np.zeros(N2)
Bias_appx_opt = np.zeros(N2)
Bias_bias_opt = np.zeros(N2)
Bias_var = np.zeros(N2)
LR_appx_opt = np.zeros(N2)
LR_bias_opt = np.zeros(N2)
LR_var = np.zeros(N2)
Zeta3 = np.zeros(N2)
BiaslinR = np.zeros(N2)
VM = np.zeros(N2)
BNS = np.zeros(N2)
LR_lin = np.zeros(N2)
LR_linR = np.zeros(N2)


k =  1.0
kf = 1.0/16.0
for i in range(0,N2):
    t_init = 0
    t_end  = 1.25**(i-12.0)
    N      = int(1000*t_end) ### Compute N grid points
    dt     = float(t_end - t_init) / N
    ts     = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(ts[-1])
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    aR = (k**(-0.5)-kf**(-0.5))/(ts[-1])
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    k_lin = k-c*ts
    k_linR = kf+c*ts
    kdotlin = np.gradient(k-c*ts,ts)
    kdotappx = np.gradient(k_appx_opt,ts)

    Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
    k_bias_opt= -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
    Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
    k_bias_optR= -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )

    kdotbiasopt = np.gradient(k_bias_opt,ts)
    kdotbiasoptR = np.gradient(k_bias_optR,ts)

    Cvar = (1.0/ts[-1])*np.log(kf/k)
    k_var = k*np.exp(Cvar*ts)
    Cvar = (1.0/ts[-1])*np.log(k/kf)
    k_varR = kf*np.exp(Cvar*ts)

    ks = 1.0*(k-c*ts)#1.0*k_appx_opt#1.0*(k-c*ts)
    ksR = 1.0*(kf+c*ts)#1.0*k_appx_optR#1.0*(kf+c*ts)
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    F = 0.5*np.log(kf/k)

    ##### Integration #####
    def fun1(t, y): return -2.0 * 1.0*ks[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun1, [ts[0], ts[-1]], [(1.0/ks[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun2(t, y): return -4.0 * 1.0*ks[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun2, [ts[0], ts[-1]], [(3.0/ks[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tF = 0.5*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_t,ts, initial=0)

    def fun3(t, y): return 2.0*W_tF[int(t/dt)] + 0.5*kdot[int(t/dt)]*xquart_t[int(t/dt)]  -2.0*ks[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun3, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun4(t, y): return kdot[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdFt = (solve_ivp(fun4, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdF = np.trapz(np.gradient(ks,ts)*xsqrd_w_t,ts)

    ##### Reverse #####
    def fun5(t, y): return -2.0 * 1.0*ksR[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun5, [ts[0], ts[-1]], [(1.0/ksR[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun6(t, y): return -4.0 * 1.0*ksR[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun6, [ts[0], ts[-1]], [(3.0/ksR[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tR = 0.5*integrate.cumtrapz(kdotR*x_sqrd_t,ts, initial=0)

    def fun7(t, y): return 2.0*W_tR[int(t/dt)] + 0.5*kdotR[int(t/dt)]*xquart_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun7, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun8(t, y): return kdotR[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdRt = (solve_ivp(fun8, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    #W_sqrdRt = integrate.cumtrapz(c*xsqrd_w_t,ts,initial=0)
    W_sqrdR = np.trapz(kdotR*xsqrd_w_t,ts)

    tau = 1.0/(2.0*kf)
    dt_tau[i] = Deltat/tau
    Wex_F_lin[i] = W_tF[-1]-F
    Wex_R_lin[i] = W_tR[-1]+F
    dW_sqrdF_lin[i] = W_sqrdF-W_tF[-1]**2.0#(W_sqrdF-W_tF[-1]**2)+(W_tF[-1]-F)**2.0
    dW_sqrdR_lin[i] = W_sqrdR-W_tR[-1]**2.0

#LR Protocol
for i in range(0,N2):
    t_init = 0
    t_end  = 1.25**(i-12.0)
    N      = int(1000*t_end) ### Compute N grid points
    dt     = float(t_end - t_init) / N
    ts    = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(ts[-1])
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    aR = (k**(-0.5)-kf**(-0.5))/(ts[-1])
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    k_lin = k-c*ts
    k_linR = kf+c*ts
    kdotlin = np.gradient(k-c*ts,ts)
    kdotappx = np.gradient(k_appx_opt,ts)

    Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
    k_bias_opt= -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
    Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
    k_bias_optR= -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )

    kdotbiasopt = np.gradient(k_bias_opt,ts)
    kdotbiasoptR = np.gradient(k_bias_optR,ts)

    Cvar = (1.0/ts[-1])*np.log(kf/k)
    k_var = k*np.exp(Cvar*ts)
    Cvar = (1.0/ts[-1])*np.log(k/kf)
    k_varR = kf*np.exp(Cvar*ts)

    ks = 1.0*k_appx_opt
    ksR = 1.0*k_appx_optR
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    F = 0.5*np.log(kf/k)

    ##### Integration #####
    def fun1(t, y): return -2.0 * 1.0*ks[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun1, [ts[0], ts[-1]], [(1.0/ks[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun2(t, y): return -4.0 * 1.0*ks[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun2, [ts[0], ts[-1]], [(3.0/ks[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tF = 0.5*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_t,ts, initial=0)

    def fun3(t, y): return 2.0*W_tF[int(t/dt)] + 0.5*kdot[int(t/dt)]*xquart_t[int(t/dt)]  -2.0*ks[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun3, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun4(t, y): return kdot[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdFt = (solve_ivp(fun4, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdF = np.trapz(np.gradient(ks,ts)*xsqrd_w_t,ts)

    ##### Reverse #####
    def fun5(t, y): return -2.0 * 1.0*ksR[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun5, [ts[0], ts[-1]], [(1.0/ksR[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun6(t, y): return -4.0 * 1.0*ksR[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun6, [ts[0], ts[-1]], [(3.0/ksR[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tR = 0.5*integrate.cumtrapz(kdotR*x_sqrd_t,ts, initial=0)

    def fun7(t, y): return 2.0*W_tR[int(t/dt)] + 0.5*kdotR[int(t/dt)]*xquart_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun7, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun8(t, y): return kdotR[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdRt = (solve_ivp(fun8, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdR = np.trapz(kdotR*xsqrd_w_t,ts)

    tau = 1.0/(2.0*kf)
    dt_tau[i] = Deltat/tau
    Wex_F_LR[i] = W_tF[-1]-F
    Wex_R_LR[i] = W_tR[-1]+F
    dW_sqrdF_LR[i] = W_sqrdF-W_tF[-1]**2.0
    dW_sqrdR_LR[i] = W_sqrdR-W_tR[-1]**2.0

#Bias Optimal Protocol
for i in range(0,N2):
    t_init = 0
    t_end  = 1.25**(i-12.0)
    N      = int(1000*t_end) ### Compute N grid points
    dt     = float(t_end - t_init) / N
    ts    = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(ts[-1])
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    aR = (k**(-0.5)-kf**(-0.5))/(ts[-1])
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    k_lin = k-c*ts
    k_linR = kf+c*ts
    kdotlin = np.gradient(k-c*ts,ts)
    kdotappx = np.gradient(k_appx_opt,ts)

    Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
    k_bias_opt = -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
    Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
    k_bias_optR = -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )

    kdotbiasopt = np.gradient(k_bias_opt,ts)
    kdotbiasoptR = np.gradient(k_bias_optR,ts)

    Cvar = (1.0/ts[-1])*np.log(kf/k)
    k_var = k*np.exp(Cvar*ts)
    Cvar = (1.0/ts[-1])*np.log(k/kf)
    k_varR = kf*np.exp(Cvar*ts)

    ks = 1.0*k_bias_opt
    ksR = 1.0*k_bias_optR
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    F = 0.5*np.log(kf/k)

    ##### Integration #####
    def fun1(t, y): return -2.0 * 1.0*ks[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun1, [ts[0], ts[-1]], [(1.0/ks[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun2(t, y): return -4.0 * 1.0*ks[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun2, [ts[0], ts[-1]], [(3.0/ks[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tF = 0.5*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_t,ts, initial=0)

    def fun3(t, y): return 2.0*W_tF[int(t/dt)] + 0.5*kdot[int(t/dt)]*xquart_t[int(t/dt)]  -2.0*ks[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun3, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun4(t, y): return kdot[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdFt = (solve_ivp(fun4, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdF = np.trapz(np.gradient(ks,ts)*xsqrd_w_t,ts)

    ##### Reverse #####
    def fun5(t, y): return -2.0 * 1.0*ksR[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun5, [ts[0], ts[-1]], [(1.0/ksR[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun6(t, y): return -4.0 * 1.0*ksR[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun6, [ts[0], ts[-1]], [(3.0/ksR[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tR = 0.5*integrate.cumtrapz(kdotR*x_sqrd_t,ts, initial=0)

    def fun7(t, y): return 2.0*W_tR[int(t/dt)] + 0.5*kdotR[int(t/dt)]*xquart_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun7, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun8(t, y): return kdotR[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdRt = (solve_ivp(fun8, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdR = np.trapz(kdotR*xsqrd_w_t,ts)

    tau = 1.0/(2.0*kf)
    dt_tau[i] = Deltat/tau
    Wex_F_Biasopt[i] = W_tF[-1]-F
    Wex_R_Biasopt[i] = W_tR[-1]+F
    dW_sqrdF_Biasopt[i] = W_sqrdF-W_tF[-1]**2.0
    dW_sqrdR_Biasopt[i] = W_sqrdR-W_tR[-1]**2.0

#Fisher information Protocol
for i in range(0,N2):
    t_init = 0
    t_end  = 1.25**(i-12.0)
    N      = int(1000*t_end) ### Compute N grid points
    dt     = float(t_end - t_init) / N
    ts    = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(ts[-1])
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    aR = (k**(-0.5)-kf**(-0.5))/(ts[-1])
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    k_lin = k-c*ts
    k_linR = kf+c*ts
    kdotlin = np.gradient(k-c*ts,ts)
    kdotappx = np.gradient(k_appx_opt,ts)

    Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
    k_bias_opt= -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
    Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
    k_bias_optR= -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )

    kdotbiasopt = np.gradient(k_bias_opt,ts)
    kdotbiasoptR = np.gradient(k_bias_optR,ts)

    Cvar = (1.0/ts[-1])*np.log(kf/k)
    k_var = k*np.exp(Cvar*ts)
    Cvar = (1.0/ts[-1])*np.log(k/kf)
    k_varR = kf*np.exp(Cvar*ts)

    ks = 1.0*k_var
    ksR = 1.0*k_varR
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    F = 0.5*np.log(kf/k)

    ##### Integration #####
    def fun1(t, y): return -2.0 * 1.0*ks[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun1, [ts[0], ts[-1]], [(1.0/ks[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun2(t, y): return -4.0 * 1.0*ks[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun2, [ts[0], ts[-1]], [(3.0/ks[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tF = 0.5*integrate.cumtrapz(np.gradient(ks,ts)*x_sqrd_t,ts, initial=0)

    def fun3(t, y): return 2.0*W_tF[int(t/dt)] + 0.5*kdot[int(t/dt)]*xquart_t[int(t/dt)]  -2.0*ks[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun3, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun4(t, y): return kdot[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdFt = (solve_ivp(fun4, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdF = np.trapz(np.gradient(ks,ts)*xsqrd_w_t,ts)

    ##### Reverse #####
    def fun5(t, y): return -2.0 * 1.0*ksR[int(t/dt)] * y + 2.0
    x_sqrd_t = (solve_ivp(fun5, [ts[0], ts[-1]], [(1.0/ksR[0])],method = 'Radau',t_eval=ts).y)[0]

    def fun6(t, y): return -4.0 * 1.0*ksR[int(t/dt)] * y + 12.0 * x_sqrd_t[int(t/dt)]
    xquart_t = (solve_ivp(fun6, [ts[0], ts[-1]], [(3.0/ksR[0]**2.0)],method = 'Radau',t_eval=ts).y)[0]

    W_tR = 0.5*integrate.cumtrapz(kdotR*x_sqrd_t,ts, initial=0)

    def fun7(t, y): return 2.0*W_tR[int(t/dt)] + 0.5*kdotR[int(t/dt)]*xquart_t[int(t/dt)] - 2.0*ksR[int(t/dt)]*y
    xsqrd_w_t = (solve_ivp(fun7, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    def fun8(t, y): return kdotR[int(t/dt)]*xsqrd_w_t[int(t/dt)]+0*y
    W_sqrdRt = (solve_ivp(fun8, [ts[0], ts[-1]], [0],method = 'Radau',t_eval=ts).y)[0]

    W_sqrdR = np.trapz(kdotR*xsqrd_w_t,ts)

    tau = 1.0/(2.0*kf)
    dt_tau[i] = Deltat/tau
    Wex_F_FI[i] = W_tF[-1]-F
    Wex_R_FI[i] = W_tR[-1]+F
    dW_sqrdF_FI[i] = W_sqrdF-W_tF[-1]**2.0
    dW_sqrdR_FI[i] = W_sqrdR-W_tR[-1]**2.0

#### Bias Estimate ####
for i in range(0,N2):
    t_init = 0
    t_end  = 1.25**(i-12.0)
    N      = int(1000*t_end) ### Compute 100 grid points
    zeta = np.zeros(N)
    ts = np.linspace(0,t_end,N+1)
    dt     = ts[1]-ts[0]
    Deltat = 1.0*t_end
    c = (k-kf)/(ts[-1])
    a = (kf**(-0.5)-k**(-0.5))/(dt*N)
    b = k**(-0.5)
    k_appx_opt = 1.0/(a*ts+b)**2
    ks = np.float64(1.0*(k-c*ts))
    aR = (k**(-0.5)-kf**(-0.5))/(dt*N)
    bR = kf**(-0.5)
    k_appx_optR = 1/(aR*ts+bR)**2
    ksR = np.float64(1.0*(kf+c*ts))
    k_lin = k-c*ts
    kdot = np.gradient(ks,ts)
    kdotR = np.gradient(ksR,ts)
    kddot = np.gradient(kdot,ts)
    F = 0.5*np.log(kf/k)

    Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
    k_bias_opt= -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
    Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
    k_bias_optR= -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )


    Cvar = (1.0/ts[-1])*np.log(kf/k)
    k_var = k*np.exp(Cvar*ts)
    Cvar = (1.0/ts[-1])*np.log(k/kf)
    k_varR = kf*np.exp(Cvar*ts)

    Biaslin[i] = -(1.0/2.0)*np.trapz(kdot**3/ks**5,ts)
    Bias_appx_opt[i] = -(1.0/2.0)*np.trapz((np.gradient(k_appx_opt,ts))**3/k_appx_opt**5,ts)
    Bias_bias_opt[i] = -(1.0/2.0)*np.trapz((np.gradient(k_bias_opt,ts))**3/k_bias_opt**5,ts)
    Bias_var[i] = -(1.0/2.0)*np.trapz((np.gradient(k_var,ts))**3/k_var**5,ts)
    LR_lin[i] = (1.0/4.0)*np.trapz(kdot**2/ks**3,ts)
    LR_appx_opt[i] = -(1.0/4.0)*np.trapz((np.gradient(k_appx_opt,ts))**2/k_appx_opt**3,ts)
    LR_bias_opt[i] = -(1.0/4.0)*np.trapz((np.gradient(k_bias_opt,ts))**2/k_bias_opt**3,ts)
    LR_var[i] = -(1.0/4.0)*np.trapz((np.gradient(k_var,ts))**2/k_var**3,ts)

plt.figure(0)
plt.plot(ts/t_end,k_lin,'k',label = 'Naive',linewidth = 3.0)
plt.plot(ts/t_end,k_var,'g:',label = 'Force Variance',linewidth = 3.0)
plt.plot(ts/t_end,(k_appx_opt),'b',label = 'Precise',linewidth = 3.0)
plt.plot(ts/ts[-1],k_bias_opt,'r--',label = 'Accurate',linewidth = 3.0)
plt.legend(frameon=False,fontsize = 14)
plt.ylabel('$k/k_{\\rm i}$',fontsize = 18)
plt.xlabel('$t/\\Delta t$',fontsize = 18)
plt.xlim(0,1)
plt.xticks([0,0.5,1],['$0$','$0.5$','$1$'],fontsize = 12)
plt.tight_layout()
#plt.savefig("protocol.pdf")

plt.figure(1)
plt.subplot(2, 2, 2)
plt.loglog(dt_tau**(1.0),0.5*np.abs(Wex_F_lin-Wex_R_lin),'k',label = 'Naive',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(Wex_F_FI-Wex_R_FI) ,'g:',label = 'Fisher Information',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(Wex_F_LR-Wex_R_LR) ,'b',label = 'Precise',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(Wex_F_Biasopt-Wex_R_Biasopt) ,'r--',label = 'Accurate',linewidth = 3.0)
plt.xlim(0.01,100)
plt.ylim(10**(-4.0),10**(1.0))
plt.xticks([0.01,1,100],fontsize = 12)
plt.yticks([0.0001,0.01,1],fontsize = 12)
plt.ylabel('bias',fontsize = 14)
plt.tight_layout()

plt.subplot(2, 2, 1)
plt.loglog(dt_tau**(1.0),0.5*np.abs(dW_sqrdF_lin+dW_sqrdR_lin),'k',label = 'naive',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(dW_sqrdF_FI+dW_sqrdR_FI) ,'g:',label = 'force variance',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(dW_sqrdF_LR+dW_sqrdR_LR) ,'b',label = 'precise',linewidth = 3.0)
plt.plot(dt_tau**(1.0),0.5*np.abs(dW_sqrdF_Biasopt+dW_sqrdR_Biasopt) ,'r--',label = 'accurate',linewidth = 3.0)
plt.legend(frameon=False)
plt.plot(dt_tau**(1.0),0.5*np.abs(dW_sqrdF_FI+dW_sqrdR_FI) ,'g:',label = 'force variance',linewidth = 3.0)
plt.xlim(0.01,1000)
plt.ylim(10**(-3.0),10**(2.0))
plt.xticks([0.01,1,100],fontsize = 12)
plt.yticks([0.01,1,100],fontsize = 12)
plt.ylabel('variance (var)',fontsize = 14)
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.loglog(dt_tau[0:43]**(1.0),(np.abs(Wex_F_lin[0:43]-Wex_R_lin[0:43])/np.abs(Wex_F_LR[0:43]-Wex_R_LR[0:43]))**(-1.0) ,'b',label = 'Precise',linewidth = 3.0)
plt.gca().set_yscale("log", base=2)
plt.plot(dt_tau[0:43]**(1.0),(np.abs(Wex_F_lin[0:43]-Wex_R_lin[0:43])/np.abs(Wex_F_Biasopt[0:43]-Wex_R_Biasopt[0:43]))**(-1.0) ,'r--',label = 'Accurate',linewidth = 3.0)
plt.plot(dt_tau[0:43]**(1.0),(np.abs(Wex_F_lin[0:43]-Wex_R_lin[0:43])/np.abs(Wex_F_FI[0:43]-Wex_R_FI[0:43]))**(-1.0)  ,'g:',label = 'Force Variance',linewidth = 3.0)
plt.plot(dt_tau**(1.0),(np.abs(Biaslin)/np.abs(Bias_appx_opt))**(-1.0) ,'b-.',linewidth = 2.0)
plt.plot(dt_tau**(1.0),(np.abs(Biaslin)/np.abs(Bias_bias_opt))**(-1.0) ,'r-.',linewidth = 2.0)
plt.plot(dt_tau**(1.0),(np.abs(Biaslin)/np.abs(Bias_var))**(-1.0) ,'g-.',linewidth = 2.0)
plt.plot([0.001,1000],[1.0,1.0],'k-.',linewidth = 2.0)
plt.xlim(0.01,100)
plt.ylim(1/32.0,2)
plt.xticks([0.01,1,100],fontsize=12)
plt.yticks([1.0/32.0,1.0/8.0,1.0/2.0,2.0],['$1/32$','$1/8$','$1/2$','$2$'],fontsize = 12)
plt.ylabel('${\\rm bias}_{\\rm des}~/~{\\rm bias}_{\\rm naive}$',fontsize = 14)
plt.xlabel('protocol duration',fontsize = 14)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.loglog(dt_tau**(1.0),(np.abs(dW_sqrdF_lin+dW_sqrdR_lin)/np.abs(dW_sqrdF_LR+dW_sqrdR_LR))**(-1.0) ,'b',label = 'Precise',linewidth = 3.0)
plt.gca().set_yscale("log", base=2)
plt.plot(dt_tau**(1.0),(np.abs(dW_sqrdF_lin+dW_sqrdR_lin)/np.abs(dW_sqrdF_Biasopt+dW_sqrdR_Biasopt))**(-1.0) ,'r--',label = 'Accurate',linewidth = 3.0)
plt.plot(dt_tau**(1.0),(np.abs(dW_sqrdF_lin+dW_sqrdR_lin)/np.abs(dW_sqrdF_FI+dW_sqrdR_FI))**(-1.0) ,'g:',label = 'Force Variance',linewidth = 3.0)
plt.plot(dt_tau**(1.0),(np.abs(LR_lin)/np.abs(LR_appx_opt))**(-1.0) ,'b-.',linewidth = 2.0)
plt.plot(dt_tau**(1.0),(np.abs(LR_lin)/np.abs(LR_bias_opt))**(-1.0) ,'r-.',linewidth = 2.0)
plt.plot(dt_tau**(1.0),(np.abs(LR_lin)/np.abs(LR_var))**(-1.0) ,'g-.',linewidth = 2.0)
plt.plot([0.001,1000],[1.0,1.0],'k-.',linewidth = 2.0)
plt.xlim(0.01,1000)
plt.ylim(1.0/4.0,2)
plt.xticks([0.01,1,100],fontsize = 12)
plt.yticks([1.0/4.0,1.0/2.0,1.0/1.0,2.0],['$1/4$','$1/2$','$1$','$2$'],fontsize = 12)
plt.ylabel('${\\rm var}_{\\rm des}~/~{\\rm var}_{\\rm naive}$',fontsize = 14)
plt.xlabel('protocol duration',fontsize = 14)
plt.tight_layout()
#plt.savefig("des_naive.pdf")

print(datetime.now()-startTime)
plt.show()