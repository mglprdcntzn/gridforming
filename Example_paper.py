import numpy as np
import matplotlib.pyplot as plt
import time

import circuit_fun as ct
import loads_lib as lds
import time_fun as tm

from scipy.linalg import expm

import random

########################################################
#Circuit generator
N = 50
Dmin = 250  #min distance btwn nodes
Dmax = 500  #max distance btwn nodes
AreaLength = 5500 # length [m] 
V = 13.2  #in kV
NPV   =  5#round(N/6) #nodes with voltage regulation
NPQ   = N - NPV

# nodes = ct.create_random_nodes(N, Dmin, AreaLength)
nodes = ct.create_picaron_nodes(N, Dmin, AreaLength)
lines = ct.create_random_lines(nodes, round(2.1*N),Dmax*5) #AreaLength/2
nodes, lines, NPV = ct.select_external_reg_nodes(nodes, lines, NPV)

ST = 250 #[kVA] rate trafos
S  = ST*(N-NPV)
########################################################
#loads at nodes
Shouse        = 5.5#kVA
houses        = lds.load_houses(nodes, Shouse, ST)
houses        = houses[NPV::]
load          = houses*Shouse
#generation at nodes
pv            = ct.DG_circuit(nodes, 0.45, ST/4, 0.01*ST)  #in kW
pv            = pv[NPV::]
#impendances of the circuit
Y, Y0, Y00    = ct.impendances_circuit(lines, N, NPV)
Ybase         = S / (V**2) / 1000  #divide by 1000 to obtain Ybase in Ohms
########################################################
#normalized circuit
barY    = Y / Ybase
barY0   = Y0 / Ybase
barY00  = Y00 / Ybase

barLoad = load / S
barpv   = pv / S
########################################################
ct.summary_circuit(nodes, lines,load,pv,'example_circuit',NPV)
########################################################
#define time
t0 =-0.00*24*60  #begining of time in min
tf = 1.00*24 * 60  #end of time
T  = 0.5  #simulation time step
nn = int((tf - t0) / T) + 1  #num of instants
t = np.linspace(t0, tf, nn)  #time vector in mins
########################################################
#drawing constants
custom_ticks = np.arange(0, t[-1]/60+1, 1)

custom_labels = []
for tick in custom_ticks:
    if tick==0:
        custom_labels = custom_labels + ['00:00']
    elif tick%24==12:
        custom_labels = custom_labels + ['12:00']
    elif tick%24==0:
        custom_labels = custom_labels + ['24:00']
    else:
        custom_labels = custom_labels + ['']
        
########################################################
#NRI0 algorithm params
itmax = 25
prec = 0.00001
########################################################
#load DG and load profiles
PVmodel = tm.read_pv_profile()
fpPV = 0.99
########################################################
#prefill vectors and matrices
ve        = np.zeros((NPQ, nn), dtype=complex) #load & DG nodes
vePV      = np.ones((NPV, nn), dtype=complex) # Sources!
ese       = np.zeros((NPQ, nn), dtype=complex)
bars0     = np.zeros((NPV, nn), dtype=complex)
Ppv       = np.zeros((NPQ, nn))
Pload     = np.zeros((NPQ, nn))
Qpv       = np.zeros((NPQ, nn))
Qload     = np.zeros((NPQ, nn))
fpPvVec   = np.zeros((NPQ, nn))
fpLoadVec = np.zeros((NPQ, nn))

vePV_mes  = expm(1j*0.0005*np.diag(np.random.randn(NPV)))
vePVinic  = np.diag(0.00005*np.random.randn(NPV)+1)@vePV_mes@np.ones((NPV, 1), dtype=complex)
vePVctrl  = np.ones((NPV, 1), dtype=complex)

########################################################
#initial conditions for NR iterations on PQ nodes
R0    = np.eye(NPQ)
Phi0  = np.zeros((NPQ, NPQ))
Vinic = R0 @ expm(1j * Phi0)
hPV   = np.conj(barY0)@(vePV[:,0].reshape((NPV, 1)))
########################################################
#perfiles de carga iniciales
perfiles_carga, perfiles_residuales = lds.load_houses_profiles(houses)
perfiles_residuales_old = perfiles_residuales
########################################################
# OPEN LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Open Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#loop through time
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    if tt==0:
        kinit = kk
    print(f'\033[KProgreso: {int(np.round(100*kk/nn))}%            ',end='\r',flush=True)
    #renovar perfiles de carga cada 24hrs
    if (tt)%(24*60)==0:
        perfiles_residuales_old = perfiles_residuales
        perfiles_carga, perfiles_residuales = lds.load_houses_profiles(houses)
    ##################################
    #PV generation
    pv_gen = tm.pv_interpole(barpv, tt%(24*60), PVmodel)
    
    Ppv[:, kk] = pv_gen
    Qpv[:, kk] = tm.reactive_power(pv_gen, fpPV)
    ##################################
    #Load
    Paux, Qaux                 = lds.load_interpole(tt%(24*60), perfiles_carga, perfiles_residuales_old)
    Pload[:, kk], Qload[:, kk] = Paux/S, Qaux/S
    ##################################
    #Power balance
    Pbalance = Ppv[:, kk] - Pload[:, kk]
    Qbalance = Qpv[:, kk] - Qload[:, kk]
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    #impose voltages at PV nodes
    vePV[:, kk] = np.squeeze(vePV_mes@vePVctrl)
    V00         = np.diag(np.squeeze(vePVctrl)) 
    hPV         = np.conj(barY0)@(vePVctrl)
    ##################################
    # NR for voltage in PQ nodes
    veaux     = tm.NRI(barY, hPV, barS, Vinic, itmax, prec)
    ve[:, kk] = veaux
    #initial conditions for next iteration
    Vinic = np.diag(veaux)
    ##################################
    #power at PV nodes
    VVV = np.diag(ve[:, kk])
    V00 = np.diag(vePV[:,kk])
    eseaux =   V00@np.conj(barY00)@np.conj(V00)@np.ones((NPV, 1)) + V00@np.conj(barY0.T)@np.conj(VVV)@np.ones((NPQ, 1))
    bars0[:, kk] = np.squeeze(eseaux)


print('\033[K              \r')
# time.sleep(1)
print(' ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')    
########################################################
#draw quantities
legendnamesPV     = [f'Node {i + 1}' for i in range(NPV)]
legendnamesPVmean = np.append(legendnamesPV,'mean value')
########################################################
Pii       = np.real(np.squeeze(S*ese))
Qii       = np.imag(np.squeeze(S*ese))
# fpii      =  np.abs(np.real(np.squeeze(ese)))/np.abs(np.squeeze(ese))

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(Pii))
axes[1].plot(t / 60, np.transpose(Qii))
axes[0].set_title('Active power at non reg. nodes')
axes[1].set_title('Reactive power at non reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel(r'$P_i$ (kW)')
axes[1].set_ylabel(r'$Q_i$ (kVA)')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'open_P_PQ'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
DabsVePQ = 1000*(np.abs(ve)-1)
FaseVePQ = np.angle(ve)*180/np.pi*1000

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePQ))
axes[1].plot(t / 60, np.transpose(FaseVePQ))

axes[0].set_title('Voltage dev. at non reg. nodes')
axes[1].set_title('Voltage phases at non reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'open_V_PQ'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

########################################################
DabsVePV = 1000*(np.abs(vePV)-1)
FaseVePV = np.angle(vePV)*180/np.pi*1000
meanDabsVePV = np.mean(DabsVePV, axis=0)
meanFaseVePV = np.mean(FaseVePV, axis=0)

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePV))
axes[0].plot(t / 60, np.transpose(meanDabsVePV),linestyle=':')

axes[1].plot(t / 60, np.transpose(FaseVePV))
axes[1].plot(t / 60, np.transpose(meanFaseVePV),linestyle=':')
fig.legend(legendnamesPVmean, loc='center', ncol=NPV+1, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Voltage dev. at reg. nodes')
axes[1].set_title('Voltage phases at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'open_V_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

########################################################
P_PV = np.real(np.squeeze(S*bars0))
Q_PV = np.imag(np.squeeze(S*bars0))

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(P_PV))
axes[1].plot(t / 60, np.transpose(Q_PV))
fig.legend(legendnamesPV, loc='center', ncol=NPV, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Active power at reg. nodes')
axes[1].set_title('Reactive power at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$P_{Reg} (kW)$')
axes[1].set_ylabel(r'$Q_{Reg} (kVA)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'open_P_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

########################################################
E0000  = T*np.cumsum(np.real(np.squeeze(bars0)), axis=1)/60

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(E0000))
fig.legend(legendnamesPV, loc='center', ncol=NPV, bbox_to_anchor=(0.5, -0.1), frameon=False)
axes[0].set_title('Energy at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel(r'$E_{0} (kWhr)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'open_E_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
# CLOSED LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Closed Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#Control periods
T1   = 10 #time for Ctrl of PV nodes [min]
n1   = np.floor(T1/T)
ndev = 2

ctrlsteps1 = np.random.randint(max(1,n1-ndev),n1+ndev,NPV)
delay1     = np.random.randint(1,n1,NPV)

T1real = ctrlsteps1*T
########################################################
#Control gains
ka = -0.001*np.ones((NPV, 1))
mm = np.ones((NPV,1))/NPV

#create a communication graph
num_edges       = int(min(NPV+2, NPV*(NPV-1)/2))
DDD             = np.zeros((num_edges,NPV), dtype=int)
TTT             = np.zeros((NPV-1,NPV), dtype=int)
edges           = set()
available_nodes = list(range(NPV))
np.random.shuffle(available_nodes)
for i in range(NPV - 1):
        u, v = available_nodes[i], available_nodes[i + 1]
        edges.add((u, v))
for i, (u, v) in enumerate(edges):
        TTT[i, u] =  1
        TTT[i, v] = -1
Tpinv = np.linalg.pinv(TTT)
while len(edges) < num_edges:
        u, v = np.random.choice(NPV, 2, replace=False)
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
for i, (u, v) in enumerate(edges):
        DDD[i, u] =  1
        DDD[i, v] = -1
#create Laplacian matrix from graph in DDD
ele   = 75e-7 #3e-7
ELE   = -(ele/NPV)*(DDD.T)@DDD

noedges = set()
for ii in range(NPV):
    for jj in range(ii+1,NPV):
        if (ii, jj) not in edges and (jj, ii) not in edges:
            noedges.add((ii,jj))
########################################################
#Draw communication graph
fig, ax = plt.subplots(figsize=(10 * 2 / 2.54, 10 * 2 / 2.54))
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
#nodes
ax.scatter(nodes[0:NPV, 0], nodes[0:NPV, 1], color='blue',zorder=3)
ax.scatter(nodes[NPV:,0], nodes[NPV:, 1], color='lightgray',zorder=1)
ax.set_aspect('equal', adjustable='box')
#labels
dist = 250
[x0, y0] = np.mean(nodes, axis=0)
for i in range(NPV):
    theta = np.arctan2( y0 - nodes[i,1] , x0 - nodes[i,0] ) + np.pi
    Dx = dist*np.cos(theta)
    Dy = dist*np.sin(theta)
    
    xx = nodes[i, 0]+Dx
    yy = nodes[i, 1]+Dy
    
    ax.text(
        xx,  # X-coordinate
        yy,  # Y-coordinate
        f"{i + 1}",  # Text (Node number)
        fontsize=10,
        ha='center',
        va='center',
        color='blue',
        zorder=3
        )
#positive communication lines
for i, (u, v) in enumerate(edges):
    ax.plot([nodes[u,0],nodes[v,0]], [nodes[u,1],nodes[v,1]], linestyle='-', color='black',zorder=2)
#positive communication lines
for i, (u, v) in enumerate(noedges):
    ax.plot([nodes[u,0],nodes[v,0]], [nodes[u,1],nodes[v,1]], linestyle=':', color='black',zorder=2)
#electric lines in background
for ll in lines:
    ii = int(ll[0])
    jj = int(ll[1])
    ax.plot([nodes[ii,0],nodes[jj,0]], [nodes[ii,1],nodes[jj,1]], linestyle='-', color='lightgray',zorder=1)
#legend
left, right = ax.get_xlim()
lower,upper = ax.get_ylim()
scale0x = left  + 100
scale0y = lower - 100
ax.plot([scale0x,scale0x+200], [scale0y,scale0y], linestyle='-', color='black')
ax.text(scale0x+250, scale0y, r'$w_{(i,j)}>0$'  , fontsize=15, ha='left', va='center',color='black') 
scale0y = scale0y - 250
ax.plot([scale0x,scale0x+200], [scale0y,scale0y], linestyle=':', color='black')
ax.text(scale0x+250, scale0y, r'$w_{(i,j)}=0$'  , fontsize=15, ha='left', va='center',color='black') 
#save figure
file_name = 'closed_communication_graph'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
#prefill new vectors and matrices
ve          = np.zeros((NPQ, nn), dtype=complex) #load & DG nodes
vePV        = np.zeros((NPV, nn), dtype=complex) # Sources!
# vePVctrl    = np.ones((NPV, 1), dtype=complex) #expm(1j * np.diag( np.random.normal(0,0.00001,NPV)))   @ (np.random.normal(1,0.000001,NPV) .reshape(NPV,1))
vePVctrl    = vePVinic #np.ones((NPV, 1), dtype=complex)

vePVabs2com = np.abs(vePVctrl)**2

ese         = np.zeros((NPQ, nn), dtype=complex)
bars0       = np.zeros((NPV, nn), dtype=complex)
bars0com    = bars0[:,0]

vememory    = np.ones((NPV, 1), dtype=complex)
Smemory     = np.zeros((NPV, 1), dtype=complex)

hPV         = np.conj(barY0)@(vePVctrl)

maxrealeigMenu = np.zeros((nn,1))
maxrealeigMu   = np.zeros((nn,1))
realbkbk       = np.zeros((nn,1))
bknom          = np.zeros((nn,1))
########################################################
#loop through time
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    if tt==0:
        kinit = kk
    print(f'\033[KProgreso: {int(np.round(100*kk/nn))}%            ',end='\r',flush=True)
    ##################################
    #Power balance
    Pbalance = Ppv[:, kk] - Pload[:, kk]
    Qbalance = Qpv[:, kk] - Qload[:, kk]
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    #impose voltages at PV nodes
    vePV[:, kk] = np.squeeze(vePV_mes@vePVctrl)
    V00         = np.diag(np.squeeze(vePVctrl)) 
    hPV         = np.conj(barY0)@(vePVctrl)
    ##################################
    # NR for voltage in PQ nodes
    veaux     = tm.NRI(barY, hPV, barS, Vinic, itmax, prec)
    ve[:, kk] = veaux
    VVV       = np.diag(veaux)
    Vinic     = np.diag(veaux)#initial conditions for next iteration
    ##################################
    #power at PV nodes
    eseaux =   V00@np.conj(barY00)@np.conj(V00)@np.ones((NPV, 1)) + V00@np.conj(barY0.T)@np.conj(VVV)@np.ones((NPQ, 1))
    bars0[:, kk] = np.squeeze(eseaux)
    ##################################
    #matrices and eigenvalues
    SSS0 = np.diag( bars0[:, kk])
    AAA0 = np.linalg.inv(V00)@SSS0
    BBB0 = V00@np.conj(barY00)
    FFF0 = np.linalg.inv(BBB0)@AAA0
    MMM0 = np.eye(barY00.shape[1]) - FFF0@np.conj(FFF0)
    HHH0 = np.linalg.inv(MMM0)@np.linalg.inv(BBB0)
    bbbT = mm.T@(V00 - np.conj(V00)@np.conj(FFF0)  )@HHH0
    
    ZZZ  = np.zeros((NPV-1,NPV-1))
    
    Menu = np.block( [ [TTT@ELE@Tpinv, ZZZ, TTT@ka],
                        [ZZZ,TTT@np.conj(ELE)@Tpinv, TTT@np.conj(ka)],
                        [bbbT@ELE@Tpinv, np.conj(bbbT)@np.conj(ELE)@Tpinv, bbbT@ka + np.conj(bbbT@ka)]])
 
    maxrealeigMenu[kk] = np.max(np.real(np.linalg.eigvals(Menu)))
    
    Mu   = np.block( [ [ELE+ka@bbbT, np.conj(ka@bbbT) ],
                        [ka@bbbT, np.conj(ELE + ka@bbbT)]])
    
    maxrealeigMu[kk]   = np.max(np.real(np.linalg.eigvals(Mu)))
    
    bkbk         = bbbT@ka + np.conj(bbbT@ka)
    realbkbk[kk] = np.real(np.squeeze(bkbk))
    
    ##################################
    #Control voltages at PV nodes
    for ii in range(NPV):          
        ###############################
        #control primario
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        if rr1 == 0:
            if tt<-1.0*24*60:
                vePVctrl[ii] = 1
            else:
                ##############################################
                #memory
                Vkk             = vememory[ii]
                Vkkabs          = abs(Vkk)
                Skk             = Smemory[ii]
                ##############################################
                #MediciOn
                vememory[ii]   = np.conj(vePV_mes[ii,ii])*vePV[ii, kk]
                Smemory[ii]    = -bars0[ii, kk] #injected or consumed?
                ##############################################
                #ComunicaciOn S
                bars0com[ii]    = Smemory[ii]
                ##############################################
                #AcciOn u
                nu              = mm.T@(vePVabs2com ) - 1 
                
                uuii            = ELE[ii,:]@bars0com + ka[ii]*nu
                ##############################################
                #funciOn f            
                Yii             = barY00[ii,ii]
                Mii             = 1 - abs(Skk)*(Vkkabs**-4)*(abs(Yii)**-2)
                Hii             = Vkk*np.conj(Yii)*uuii/Mii
                
                fii             = -(np.conj(Skk)/((np.conj(Vkk)**2)*Yii) )*Hii + np.conj(Hii)
                ##############################################
                #actualizar ctrl
                vePVctrl[ii]    = Vkk + T1real[ii]*fii 
            ##############################################
            #ComunicaciOn V
            vePVabs2com[ii] = abs(vePVctrl[ii])**2
 
print('\033[K              \r')
# time.sleep(1)
print(' ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
DabsVePQ = 1000*(np.abs(ve)-1)
FaseVePQ = np.angle(ve)*180/np.pi*1000

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePQ))
axes[1].plot(t / 60, np.transpose(FaseVePQ))

axes[0].set_title('Voltage dev. at non reg. nodes')
axes[1].set_title('Voltage phases at non reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_V_PQ'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

########################################################
DabsVePV = 1000*(np.abs(vePV)-1)
FaseVePV = np.angle(vePV)*180/np.pi*1000
meanDabsVePV = np.mean(DabsVePV, axis=0)
meanFaseVePV = np.mean(FaseVePV, axis=0)

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePV))
axes[0].plot(t / 60, np.transpose(meanDabsVePV),linestyle=':')
axes[1].plot(t / 60, np.transpose(FaseVePV))
axes[1].plot(t / 60, np.transpose(meanFaseVePV),linestyle=':')

fig.legend(legendnamesPVmean, loc='center', ncol=NPV+1, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Voltage dev. at reg. nodes')
axes[1].set_title('Voltage phases at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_V_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
P_PV = np.real(np.squeeze(S*bars0))
Q_PV = np.imag(np.squeeze(S*bars0))

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(P_PV))
axes[1].plot(t / 60, np.transpose(Q_PV))

fig.legend(legendnamesPV, loc='center', ncol=NPV, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Active power at reg. nodes')
axes[1].set_title('Reactive power at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$P_{Reg} (kW)$')
axes[1].set_ylabel(r'$Q_{Reg} (kVA)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_P_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
E0000  = T*np.cumsum(np.real(np.squeeze(bars0)), axis=1)/60

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(E0000))
fig.legend(legendnamesPV, loc='center', ncol=NPV, bbox_to_anchor=(0.5, -0.1), frameon=False)
axes[0].set_title('Energy at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel(r'$E_{0} (kWhr)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_E_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
ee = TTT@np.squeeze(S*bars0)
abseeP = np.linalg.norm(np.real(ee), axis=0)
abseeQ = np.linalg.norm(np.imag(ee), axis=0)
fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(abseeP))
axes[1].plot(t / 60, np.transpose(abseeQ))

axes[0].set_title(r'Norm of active power error')
axes[1].set_title(r'Norm of reactive power error')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\|real\{e\}\|  (kW)$')
axes[1].set_ylabel(r'$\|imag\{e\}\| (kVA)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_ee_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
absVPV = np.abs(vePV)**2 - 1
nuu    = mm.T@absVPV
maxnu  = np.max(absVPV,axis=0)
minnu  = np.min(absVPV,axis=0)

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(nuu))
axes[0].plot(t / 60, np.transpose(maxnu),linestyle=':', linewidth=1.00)
axes[0].plot(t / 60, np.transpose(minnu),linestyle=':', linewidth=1.00)
axes[1].plot(t / 60, np.squeeze(1000000*realbkbk))

axes[0].set_title(r'Dev. of quadratic reg. voltages')
axes[1].set_title(r'Dev. eigenvalue $\times 10^6$')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\nu$')

axes[0].legend([r'$\nu$', r'$max\{ V_0V_0^*-I\}$', r'$min\{ V_0V_0^*-I\}$'], 
               loc='lower center',  # Places legend inside, at the lower center
               ncol=3,  # Arranges items in a single horizontal row
               frameon=True)  # Optional: Adds a frame around the legend

# axes[0].set_ylim(np.min(nuu), np.max(nuu))

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'closed_nu_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
# DROOP CTRL
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Droop Control Closed Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#Droop gains
DD = 0.0001*np.ones((NPV, 1))
########################################################
#prefill new vectors and matrices
ve          = np.zeros((NPQ, nn), dtype=complex) #load & DG nodes
vePV        = np.zeros((NPV, nn), dtype=complex) # Sources!
vePVctrl    = vePVinic #np.ones((NPV, 1), dtype=complex)

ese         = np.zeros((NPQ, nn), dtype=complex)
bars0       = np.zeros((NPV, nn), dtype=complex)
bars0com    = bars0[:,0]

hPV         = np.conj(barY0)@(vePVctrl)

########################################################
#loop through time
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    if tt==0:
        kinit = kk
    print(f'\033[KProgreso: {int(np.round(100*kk/nn))}%            ',end='\r',flush=True)
    ##################################
    #Power balance
    Pbalance = Ppv[:, kk] - Pload[:, kk]
    Qbalance = Qpv[:, kk] - Qload[:, kk]
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    #impose voltages at PV nodes
    vePV[:, kk] = np.squeeze(vePV_mes@vePVctrl)
    V00         = np.diag(np.squeeze(vePVctrl)) 
    hPV         = np.conj(barY0)@(vePVctrl)
    ##################################
    # NR for voltage in PQ nodes
    veaux     = tm.NRI(barY, hPV, barS, Vinic, itmax, prec)
    ve[:, kk] = veaux
    VVV       = np.diag(veaux)
    Vinic     = np.diag(veaux)#initial conditions for next iteration
    ##################################
    #power at PV nodes
    eseaux =   V00@np.conj(barY00)@np.conj(V00)@np.ones((NPV, 1)) + V00@np.conj(barY0.T)@np.conj(VVV)@np.ones((NPQ, 1))
    bars0[:, kk] = np.squeeze(eseaux)
    ##################################
    #Control voltages at PV nodes
    for ii in range(NPV):          
        ###############################
        #control primario
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        if rr1 == 0:
            if tt<-1.0*24*60:
                vePVctrl[ii] = 1
            else:
                ##############################################
                #MediciOn
                Vkk             = np.conj(vePV_mes[ii,ii])*vePV[ii, kk]
                Skk             = -bars0[ii, kk] #injected or consumed?
                ##############################################
                #AcciOn u
                uuii            = -DD[ii]*Skk
                ##############################################
                #actualizar ctrl
                # vePVctrl[ii]    = Vkk + T1real[ii]*uuii
                vePVctrl[ii]    = 1 + uuii
                # vePVctrl[ii]    = uuii
print('\033[K              \r')
# time.sleep(1)
print(' ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
DabsVePQ = 1000*(np.abs(ve)-1)
FaseVePQ = np.angle(ve)*180/np.pi*1000

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePQ))
axes[1].plot(t / 60, np.transpose(FaseVePQ))

axes[0].set_title('Voltage dev. at non reg. nodes')
axes[1].set_title('Voltage phases at non reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'droop_V_PQ'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

########################################################
DabsVePV = 1000*(np.abs(vePV)-1)
FaseVePV = np.angle(vePV)*180/np.pi*1000
meanDabsVePV = np.mean(DabsVePV, axis=0)
meanFaseVePV = np.mean(FaseVePV, axis=0)

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(DabsVePV))
axes[0].plot(t / 60, np.transpose(meanDabsVePV),linestyle=':')
axes[1].plot(t / 60, np.transpose(FaseVePV))
axes[1].plot(t / 60, np.transpose(meanFaseVePV),linestyle=':')

fig.legend(legendnamesPVmean, loc='center', ncol=NPV+1, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Voltage dev. at reg. nodes')
axes[1].set_title('Voltage phases at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$\Delta V (‰)$')
axes[1].set_ylabel(r'$\angle V (10^{-3}\times 1\degree)$')

axes[0].set_ylim(np.min(DabsVePV[:,len(DabsVePV.T)//2:]), np.max(DabsVePV[:,len(DabsVePV.T)//2:]))


for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'droop_V_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
########################################################
P_PV = np.real(np.squeeze(S*bars0))
Q_PV = np.imag(np.squeeze(S*bars0))

fig, axes = plt.subplots(1,2,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(P_PV))
axes[1].plot(t / 60, np.transpose(Q_PV))

fig.legend(legendnamesPV, loc='center', ncol=NPV, bbox_to_anchor=(0.5, -0.1), frameon=False)

axes[0].set_title('Active power at reg. nodes')
axes[1].set_title('Reactive power at reg. nodes')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')

axes[0].set_ylabel(r'$P_{Reg} (kW)$')
axes[1].set_ylabel(r'$Q_{Reg} (kVA)$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Day Time (hrs)')
fig.tight_layout()

file_name = 'droop_P_PV'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')