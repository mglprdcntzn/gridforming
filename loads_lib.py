import numpy as np
import random
import math
import matplotlib.pyplot as plt
#############################################################
def load_houses(nodes, Shouse, Strafo):
    N = nodes.shape[0]
    
    nnominal = Strafo/Shouse
    nhouses = np.random.randint(nnominal*0.25, nnominal*1.25, N)
    
    return nhouses
#############################################################
def load_houses_events(eventos,Aparatos,tinic,tfin,permanente=0):
    for aparato in Aparatos:
        if permanente:
            P  = aparato[0]/1000
            fp = aparato[1]
            Q  = P * np.sqrt(np.reciprocal(fp**2) - 1)
                        
            ev = (tinic,tfin,P,Q)
            eventos.append(ev)
        else:
            prob = aparato[2]
            if np.random.random()<prob:
                P  = aparato[0]/1000
                fp = aparato[1]
                Q  = P * np.sqrt(np.reciprocal(fp**2) - 1)
                
                Tmean = aparato[3]
                t0 = np.random.uniform(tinic, tfin)
                tf = t0 + np.random.normal(Tmean,0.1)
                
                ev = (t0,tf,P,Q)
                eventos.append(ev)
                # if tf<=24*60:
                #     ev = (t0,tf,P,Q)
                #     eventos.append(ev)
                # else:
                #     ev = (t0,24*60,P,Q)
                #     eventos.append(ev)
                #     ev = (0,tf-24*60,P,Q)
                #     eventos.append(ev)
    return eventos    
#############################################################
def load_houses_profiles(houses):
    N = len(houses)
    load_profile = []
    load_residuos= []
    
    for ii in range(N):
        eventos = []
        for hh in range(houses[ii]):
            #permanentes
            Aparatos = np.array([#P,fp
                    [450, 0.80],#refri
                    [ 60, 0.99] #otros
                    ])
            eventos = load_houses_events(eventos,Aparatos,0,24*60,1)
            #0-3 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [150, 0.95, 0.10,2*60],#TV 1
                    [15*4, 0.999, 0.50,2*60],#Luces 1
                    [15*4, 0.999, 0.10,2*60],#Luces 2
                    [8, 0.80, 0.5,1*60],#celular
                    [1000, 1.00, 0.2,1*60],#calefacciOn
                    [150, 0.90,0.10,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,0,3*60,0)
            #3-6 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [15*4, 0.999, 0.10,3*60],#Luces 1
                    [150, 0.90,0.01,1*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,3*60,6*60,0)
            #6-9 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [7000, 0.95, 0.01/2, 2*60],#electric vehicle
                    [15*4, 0.999, 0.80,1*60],#Luces 1
                    [15*8, 0.999, 0.80,2*60],#Luces 2
                    [1800, 1.00, 0.80,4],#hervidor
                    [1500, 0.95, 4/7,4],#secador de pelo
                    [800, 0.90, 0.9,4],#microondas
                    [8, 0.80, 0.2,1*60],#celular
                    [1000, 1.00, 0.5,1*60],#calefacciOn
                    [150, 0.95, 0.80,1*60]#TV 1
                    ])
            eventos = load_houses_events(eventos,Aparatos,6*60,9*60,0) 
            #9-12 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [1500, 0.85, 4/7,1*60+15],#lavadora
                    [1800, 0.95, 1/7,45],#horno electrico
                    [1800, 0.85, 2/7,15],#aspiradora
                    [1800, 1.00, 0.50,4],#hervidor
                    [800, 0.90, 0.2,4],#microondas
                    [8, 0.80, 0.01,1*60],#celular
                    [150, 0.95, 0.20,1*60],#TV 1
                    [150, 0.90,0.20,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,9*60,12*60,0)    
            #12-15 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [1500, 0.85, 1/7,1*60+15],#lavadora
                    [1800, 0.95, 1/7,45],#horno electrico
                    [1800, 1.00, 0.10,4],#hervidor
                    [800, 0.90, 0.5,4],#microondas
                    [8, 0.80, 0.01,1*60],#celular
                    [150, 0.95, 0.80,2*60],#TV 1
                    [150, 0.90,0.20,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,12*60,15*60,0)          
            #15-18 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [7000, 0.95, 0.01, 2*60],#electric vehicle
                    [8, 0.80, 0.01,1*60],#celular
                    [1800, 0.85, 1/7,15],#aspiradora
                    [150, 0.95, 0.80,2*60],#TV 1
                    [15*3, 0.999, 0.80,30],#Luces 1
                    [150, 0.90,0.50,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,15*60,18*60,0)
            #18-21 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [7000, 0.95, 0.01, 2*60],#electric vehicle
                    [8, 0.80, 0.01,1*60],#celular
                    [1800, 1.00, 0.10,4],#hervidor
                    [800, 0.90, 0.5,4],#microondas
                    [150, 0.95, 0.80,2*60],#TV 1
                    [15*10, 0.999, 0.80,2*60],#Luces 1
                    [150, 0.90,0.75,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,18*60,21*60,0)
            #21-24 hrs
            Aparatos = np.array([#P,fp,prob,Tmean
                    [7000, 0.95, 0.01/2, 2*60],#electric vehicle
                    [8, 0.80, 0.5,1*60],#celular
                    [150, 0.95, 0.56,3*60],#TV 1
                    [15*10, 0.999, 0.90,2*60],#Luces 1
                    [15*5, 0.999, 0.50,2*60],#Luces 2
                    [150, 0.90,0.75,2*60] #PC
                    ])
            eventos = load_houses_events(eventos,Aparatos,21*60,24*60,0)
        #acumular eventos
        instantes = []
        for ev in eventos:
            instantes.append(ev[0])
            instantes.append(ev[1])
        instantes = list(set(instantes))
        instantes.sort()
        
        perfil = []
        residuo = []
        if houses[ii]>0:
            for tt in instantes:
                P = 0
                Q = 0
                for ev in eventos:
                    if ev[0]<=tt<=ev[1]:
                        P = P + ev[2]
                        Q = Q + ev[3]
                if tt<=24*60:
                    perfil.append((tt,P,Q))
                else:
                    residuo.append((tt-24*60,P,Q))
        else:
            perfil.append((0,0,0))
            residuo.append((-24*60,0,0))
        
        # residuo.insert(0, (0,perfil[-1][1],perfil[-1][2]))  
        # residuo.insert(0, (0,0,0))  
        
        load_profile.append(perfil)
        load_residuos.append(residuo)
    return load_profile, load_residuos
#############################################################
def load_interpole(tt, perfiles, residuos):
    hr = math.floor(tt / 60)
    while hr > 23:
        hr = hr - 24
        tt = tt - 24 * 60

    N = len(perfiles)
    P = np.zeros((N,1))
    Q = np.zeros((N,1))
    
    for ii in range(N):
        eventoanterior = perfiles[ii][0]
        for evento in perfiles[ii]:
            if tt<=evento[0]:
                P[ii] = eventoanterior[1]
                Q[ii] = eventoanterior[2]
                break
            eventoanterior = evento
        eventoanterior = residuos[ii][0]
        for evento in residuos[ii]:
            if tt<=evento[0]:
                P[ii] = P[ii] + eventoanterior[1]
                Q[ii] = Q[ii] + eventoanterior[2]
                break
            eventoanterior = evento
    
    noise = np.array([random.gauss(1, 0.01) for _ in range(N)])
    noise = noise[:,np.newaxis]

    P     = P*noise
    Q     = Q*noise
    return np.squeeze(P),np.squeeze(Q)
#############################################################