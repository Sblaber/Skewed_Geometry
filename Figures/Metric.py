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
from scipy import integrate
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from datetime import datetime
#import seaborn as sns; sns.set()
startTime = datetime.now()
print(startTime)

## This code plots fig. 1 from https://doi.org/10.1063/5.0033405 ##
## It takes seconds to run ##

t_init = 0
t_end  = 1
N      = 1000 ### Compute 100 grid points
dt     = float(t_end - t_init) / N
ts    = np.linspace(t_init,t_end,num=N)
k =  1.0
kf = 2.0
Deltat = 1.0*t_end
c = (k-kf)/(dt*N)
C2 = (1/(dt*N))*( ( (1+2*k*dt*N+k*kf*(dt*N)**2)**0.5-kf*dt*N-1 )/(2+kf*dt*N) )
C3 = (1/(dt*N))*( ( (1+2*kf*dt*N+k*kf*(dt*N)**2)**0.5-k*dt*N-1 )/(2+k*dt*N) )
k_opt = (k- C2*(1+C2*ts) )/(1+C2*ts)**2
C2 = (1/(dt*N))*( ( (1+2*kf*dt*N+kf*k*(dt*N)**2)**0.5-k*dt*N-1 )/(2+k*dt*N) )
C3 = (1/(dt*N))*( ( (1+2*k*dt*N+kf*k*(dt*N)**2)**0.5-kf*dt*N-1 )/(2+kf*dt*N) )
k_optR = (kf- C2*(1+C2*ts) )/(1+C2*ts)**2
a = (kf**(-0.5)-k**(-0.5))/(dt*N)
b = k**(-0.5)
k_appx_opt = 1/(a*ts+b)**2
a = (k**(-0.5)-kf**(-0.5))/(dt*N)
b = kf**(-0.5)
k_appx_optR = 1/(a*ts+b)**2
ks = 1.0*k_appx_opt#k-c*ts
ks[0] = 1.0*k
ks[-1] = 1.0*kf
k_lin = k-c*ts
k_linR = kf+c*ts
Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
k_bias_opt= -3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )
Cbias = -(1.0/t_end)*(3.0/2.0)*(k**(-2.0/3.0)-kf**(-2.0/3.0))
k_bias_optR= -3.0*kf**(2.0/3.0)/( (kf**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*kf**(2.0/3.0)*ts-3.0) )

Cbias = -(1.0/t_end)*(3.0/2.0)*(kf**(-2.0/3.0)-k**(-2.0/3.0))
k_bias_opt2 = 3.0*k**(2.0/3.0)/( (k**(-2.0/3.0)-2.0*Cbias*ts/3.0)**(1.0/2.0)*(2.0*Cbias*k**(2.0/3.0)*ts-3.0) )

#k_bias_opt = 5.52419/(2*ts+3.02496)**(3.0/2.0)

ks = np.linspace(kf,k,num=N)
kdot = np.linspace(-1,1,num=N)
FI = np.zeros((N,N))
PFI = np.zeros((N,N))
friction = np.zeros((N,N))
friction1 = np.zeros((N,N))
friction2 = np.zeros((N,N))
P = np.zeros((N,N))
P1 = np.zeros((N,N))
P2 = np.zeros((N,N))
Vector_field = np.zeros((N,N))
for i in range(0,N):
    FI[i,:] = 1.0/(2.0*ks**2.0)
    PFI[i,:] = kdot[i]**2.0/(2.0*ks**2.0)
    friction1[i,:] = 1.0/(4*ks**3.0)
    P1[i,:] = kdot[i]**2.0/(4*ks**3.0)
    friction2[i,:] = kdot[i]/(8.0*ks**5.0)
    P2[i,:] = kdot[i]**3.0/(8.0*ks**5.0)
    friction[i,:] = 1.0/(4*ks**3.0)+kdot[i]/(8.0*ks**5.0)
    P[i,:] = kdot[i]**2.0/(4*ks**3.0)+kdot[i]**3.0/(8.0*ks**5.0)
    Vector_field[i,:] = kdot[i]*(3.0*ks**2.0-5.0*kdot[i])/(2.0*ks**3.0-3*ks*kdot[i])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

c = ax1.pcolormesh(ks**(1.0), kdot, np.abs(FI), cmap='Reds', vmin=0, vmax=0.5)
cbar=fig.colorbar(c, ax=ax1,ticks = [0,0.25,0.5])
cbar.set_ticklabels(['$0$','$0.25$','$0.5$'])
#cbar.set_label('$\\langle \\delta f^2\\rangle_{\\lambda(t)}$',fontsize = 16)
cbar.set_label('force variance',fontsize = 12)
CS = ax1.contour(ks**(1.0), kdot, PFI,[10000*(PFI.min()),500000*(PFI.min()),1750000*(PFI.min())],cmap = 'gray',vmin =PFI.min(),vmax = PFI.min() )
#ax1.set_ylabel('$\\dot{\lambda}$',fontsize = 16)
ax1.set_ylabel('CP velocity',fontsize = 12)
ax1.set_xticks([1,1.5,2])
ax1.set_xticklabels(['$1$','$1.5$','$2$'])
ax1.set_yticks([-1,0,1])
plt.tight_layout()

c = ax2.pcolormesh(ks**(1.0), kdot, np.abs(friction1), cmap='Reds', vmin=0, vmax=0.25)
cbar=fig.colorbar(c, ax=ax2,ticks = [0,0.125,0.25])
cbar.set_ticklabels(['$0$','$0.125$','$0.25$'])
#cbar.set_label('$\zeta^{(1)}$',fontsize = 16)
cbar.set_label('Stokes\' friction',fontsize = 12)
CS = ax2.contour(ks**(1.0), kdot, P1,[10000*(P1.min()),500000*(P1.min()),2000000*(P1.min())],cmap = 'gray',vmin =P1.min(),vmax = P1.min() )
#plt.ylabel('$\\dot{\lambda}(t)$',fontsize = 24)
#plt.xlabel('$\lambda(t)$',fontsize = 24)
ax2.set_xticks([1,1.5,2])
ax2.set_xticklabels(['$1$','$1.5$','$2$'])
ax2.set_yticks([-1,0,1])
plt.tight_layout()

c = ax3.pcolormesh(ks**(1.0), kdot, friction2, cmap='RdBu_r', vmin=-0.1, vmax=0.1)#, vmin=friction2.min(), vmax=friction2.max())
cbar = fig.colorbar(c, ax=ax3,ticks = [-0.1,0,0.1])
cbar.set_ticklabels(['$-0.1$','$0$','$0.1$'])
#cbar.set_label('$\\frac{1}{4}\\dot{\lambda}\zeta^{(2)}$',fontsize = 16)
cbar.set_label('supra-Stokes\' contribution',fontsize = 12)
CS = ax3.contour(ks**(1.0), kdot, P2,[1000000*(np.abs(P2).min()),250000000*(np.abs(P2).min()),2000000000*(np.abs(P2).min())],cmap = 'gray',vmin =P2.min(),vmax = P2.min() )
CS = ax3.contour(ks**(1.0), kdot, P2,-1.0*np.array([1000000*(np.abs(P2).min()),250000000*(np.abs(P2).min()),2000000000*(np.abs(P2).min())])[::-1],cmap = 'gray',vmin =P2.min(),vmax = P2.min() )
#ax3.set_ylabel('$\\dot{\lambda}$',fontsize = 16)
ax3.set_ylabel('CP velocity',fontsize = 12)
ax3.set_xlabel('control parameter (CP)',fontsize = 12)
ax3.set_xticks([1,1.5,2])
ax3.set_xticklabels(['$1$','$1.5$','$2$'])
ax3.set_yticks([-1,0,1])
plt.tight_layout()


c = ax4.pcolormesh(ks**(1.0), kdot, friction, cmap='Reds', vmin=0, vmax=0.25)#, vmin=friction2.min(), vmax=friction2.max())
cbar = fig.colorbar(c, ax=ax4,ticks = [0,0.125,0.25])
cbar.set_ticklabels(['$0$','$0.125$','$0.25$'])
#cbar.set_label('$\zeta^{\\rm tot}$',fontsize = 16)
cbar.set_label('total friction',fontsize = 12)
CS = ax4.contour(ks**(1.0), kdot, P,[10000*(P.min()),500000*(P.min()),2000000*(P.min())],cmap = 'gray',vmin =P.min(),vmax = P.min() )
#ax4.set_xlabel('$\lambda$',fontsize = 16)
ax4.set_xlabel('control parameter (CP)',fontsize = 12)
ax4.set_xticks([1,1.5,2])
ax4.set_xticklabels(['$1$','$1.5$','$2$'])
ax4.set_yticks([-1,0,1])
plt.tight_layout()
#fig.savefig("foo.png",dpi = 800)

#fig, ax = plt.subplots()
#ax.streamplot(ks, kdot, Vector_field, (Vector_field*0+1),color='k')#,start_points=seed_points)
#ax.pcolormesh(ks, kdot, friction, cmap='RdBu', vmin=0, vmax=0.4)
#plt.plot(k_bias_opt**(1.0),np.gradient(k_bias_opt,ts),'k:',linewidth = 2.0)
#plt.plot((k_appx_opt)**(1.0),np.gradient(k_appx_opt,ts),'k--',linewidth = 2.0)

plt.figure()
plt.plot(ts/t_end,k_opt-k,'k')
plt.plot(ts/t_end,k_lin-k,'k')
plt.plot(ts/t_end,(k_appx_opt-k),'k')
plt.plot(ts/t_end,k_bias_opt-k,'k:',linewidth = 2.0)
plt.plot(0,0,'ko')
plt.plot(1,1,'ko')
plt.ylabel('$\lambda(t)$',fontsize = 16)
plt.xlabel('$t$',fontsize = 16)
plt.grid()
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)

plt.show()
