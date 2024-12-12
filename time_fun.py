import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate
#############################################################
def NRI(barY, hPV, barS, Vinic, itmax, prec):
    #Newton Raphson method for power flow
    NPQ     = Vinic.shape[0]
    unos    = np.ones((NPQ, 1))
    I       = np.eye(NPQ)
    B       = np.conj(barY)
    invbarS = np.linalg.inv(barS)
    barSvec = barS@unos
    
    F0      = B@Vinic@Vinic@invbarS
    M       = np.linalg.inv( I - F0@np.conj(F0)  ) 
    #loop
    ve      = Vinic@unos
    permiso = True
    itcont  = 0
    while permiso:
        h     = hPV + B@np.conj(ve) - (1/ve)*barSvec
        invA  = np.diag(np.squeeze((ve**2))) @ invbarS
        F     = B@np.conj(invA)
        M     = np.linalg.inv( I - F@np.conj(F)  )
        # M     = I + F@np.conj(F)+ (F@np.conj(F))**2
        # M     = I + F@np.conj(M)@np.conj(F)
            
        Delta = h - F@np.conj(h)
        ve    = ve - invA@M@Delta
      
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(h, ord=2) < prec:
            permiso = 0
            
    return  np.squeeze(ve)


#############################################################
def reactive_power(P, fp):
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(P))])
    Q = P * np.sqrt(np.reciprocal(fp**2) - 1) * noise
    return Q

#############################################################
def read_pv_profile():
    filename = 'pv_profile.csv'
    df = pd.read_csv(filename)
    model = df.values
    
    return model

#############################################################
def pv_interpole(pv_installed, tt, model):
    hr = math.floor(tt / 60)
    if hr > 23:
        hr = hr - 24
        tt = tt - 24 * 60
    
    coefs = model[:, hr]
    inst = tt / (24 * 60)
    times = np.array([[inst**3], [inst**2], [inst], [1]])
    pv_profile = coefs @ times
    
    noise = np.array([random.gauss(1, 0.002) for _ in range(len(pv_installed))])
    pv_generated = pv_installed * pv_profile * noise
    pv_generated = np.clip(pv_generated, 0, None)  #eliminate negatives
    
    return pv_generated