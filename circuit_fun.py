import numpy as np
import random
import math
import matplotlib.pyplot as plt
#############################################################
def load_circuit(nodes, LoadMean, dev):
    N = nodes.shape[0]
    
    load = np.array([random.gauss(LoadMean, dev) for _ in range(N - 1)])
    load = np.clip(load, 0, None)  #eliminate negatives
    classes = 3  #residential,industrial,commercial
    loadmix = np.random.dirichlet(np.ones(classes), N - 1)
    
    return load, loadmix
#############################################################
def DG_circuit(nodes, PVprob, PVmean, PVdev):
    N = nodes.shape[0]
    
    pv = np.array([
      random.gauss(PVmean, PVdev) *
      random.choices([0, 1], weights=[1 - PVprob, PVprob])[0]
      for _ in range(N)
    ])
    pv = np.clip(pv, 0, None)  #eliminate negatives
    
    return pv

#############################################################
def impendances_circuit(lines, N, NPV):
    ###############
    NB = lines.shape[0] #num of branches
    ###############
    Rperkm = 0.2870120
    Xperkm = 0.5508298
    Rpermt = Rperkm / 1000
    Xpermt = Xperkm / 1000
    ###############
    epsilon = 1/100
    upper = 1 + epsilon
    lower = 1- epsilon
    ###############
    W = np.zeros((NB, NB), dtype=complex)  #admitances of each line
    D = np.zeros((NB, N), dtype=int)  #incidence matrix
    
    for ll in range(NB):
        origin  = int(lines[ll, 0]) 
        destiny = int(lines[ll, 1]) 
        Rll = Rpermt
        Xll = Xpermt
        #impedance
        Z = lines[ll, 2] * ( Rll* random.uniform(lower, upper) + 1j * Xll * random.uniform(lower, upper) )
        #admittance
        W[ll, ll] = 1 / Z
        #incidences
        D[ll, origin] = -1
        D[ll, destiny] = 1
    
    #admitances matrix
    hatY = np.transpose(D) @ W @ D
    
    #partitionate adm matrx
    Y00 = hatY[0:NPV, 0:NPV]
    Y0  = hatY[NPV:N, 0:NPV]
    Y   = hatY[NPV:N, NPV:N]
    
    return Y, Y0, Y00
#############################################################
def summary_circuit(nodes, lines,load,dg,plt_name,NPV):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    N = nodes.shape[0]
    Nl = lines.shape[0]
    Sload = 10*np.ceil(load/10).sum()/1000
    Sdg = np.round(dg.sum())/1000
    length = np.round(lines[:,2].sum())/1000
    
    print('Example circuit nodes: %s' % N)
    print('Example circuit regulation nodes: %s' % NPV)
    print('Example circuit lines: %s' % Nl)
    print('Example circuit rated power (load): %s [MVA]' % Sload)
    print('Example circuit installed DG: %s [MVA]' % Sdg)
    print('Example circuit total length: %s [km]' % length)
    
    cts_data = (N,Sload,Sdg,length,NPV)
    print_circuit(nodes, lines, plt_name,cts_data,dg)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return
#############################################################
def print_circuit(nodes, lines, plt_name,cts_data,pv):
    #circuit data
    N,Sload,Sdg,length,NPV = cts_data
    #prepare figure
    fig, ax = plt.subplots(figsize=(10 * 2 / 2.54, 10 * 2 / 2.54))
    
    #ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    #nodes
    yoffset = 40
    ax.scatter(nodes[0:NPV, 0], nodes[0:NPV, 1]+yoffset, color='blue',zorder=3)
    
    for nn in range(N-NPV):
        if pv[nn]>0:
            colour = 'orange'
        else:
            colour = 'black'
        ax.scatter(nodes[nn+NPV, 0], nodes[nn+NPV, 1]+yoffset, color=colour,zorder=3)
    
    #lines
    rows, cols = lines.shape
    for ll in range(0, rows):
        xx = np.array([
          nodes[lines[ll, 0].astype(int), 0],
          nodes[lines[ll, 1].astype(int), 0]
        ])
        yy = np.array([
          nodes[lines[ll, 0].astype(int), 1]+yoffset,
          nodes[lines[ll, 1].astype(int), 1]+yoffset
        ])
        ax.plot(xx, yy, linestyle='-', color='gray',zorder=2)
    #labels on nodes
    dist = 120*np.sqrt(2)
    label_positions = []
    for ii in range(len(nodes[:, 0])):
        #decide label color
        if ii < NPV:
            color_str = 'blue'
        elif pv[ii-NPV]>0:
            color_str = 'orange'            
        else:
            color_str = 'black'
        
        #choose where to put the label
        angles = []
        for ll in range(0,rows):
            if lines[ll,0]==ii:
                jj = int(lines[ll,1])
                theta = np.arctan2( nodes[jj,1] - nodes[ii,1] , nodes[jj,0] - nodes[ii,0] )
                angles.append(theta)
                
            elif lines[ll,1]==ii:
                jj = int(lines[ll,0])
                theta = np.arctan2( nodes[jj,1] - nodes[ii,1] , nodes[jj,0] - nodes[ii,0] )
                angles.append(theta)
        
        angles.sort()
        anglediffs = []
        thant = angles[-1] - 2*np.pi
        for th in angles:
            anglediffs.append( (abs(th - thant), thant ) )
            thant = th
        
        anglediffs = sorted(anglediffs, key=lambda x: x[0], reverse=True)
        
        angles = [ (x[1]+(x[0]/2)) for x in anglediffs]

        checking = True
        th_prior = 0
        while checking:            
            theta = angles[th_prior] 
            Dx = dist*np.cos(theta)
            Dy = dist*np.sin(theta)
            
            xx = nodes[ii, 0]+Dx
            yy = nodes[ii, 1]+Dy+yoffset
            
            flag = True
            for pair in label_positions:
                xold = pair[0]
                yold = pair[1]
                if (xx-xold)**2 + (yy-yold)**2 < (4*dist/4)**2:
                    flag = False
            if flag:
                checking = False
            else:
                th_prior = th_prior +1
                
        label_positions.append((xx,yy))
        
        ax.text(xx, yy, ii+1, fontsize=10, ha='center', va='center',color=color_str,zorder=3)
    
    ax.set_aspect('equal', adjustable='box')
    #find corners coordinates
    left, right = ax.get_xlim()
    lower,upper = ax.get_ylim()
    #draw a light grid
    for x in np.arange(left+1000, right, 1000):
        ax.vlines(x, lower, upper, colors='lightgray', linestyles='--', linewidth=0.5, zorder=1)
    for y in np.arange(lower+1000, upper, 1000):
        ax.hlines(y, left, right, colors='lightgray', linestyles='--', linewidth=0.5, zorder=1)
        
    ax.vlines(left, lower, upper, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.vlines(right, lower, upper, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.hlines(lower, left, right, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.hlines(upper, left, right, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    
    #draw scale reference at lower left corner
    scale0x = left #+ 20
    scale0y = lower #- 20
    ax.plot([scale0x,scale0x+1000], [scale0y,scale0y], linestyle='-', color='black')
    ax.plot([scale0x,scale0x], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+250,scale0x+250], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+500,scale0x+500], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+750,scale0x+750], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+1000,scale0x+1000], [scale0y+50,scale0y-50], linestyle='-', color='black')
    
    ax.text(scale0x+000, scale0y-70, '0m'  , fontsize=8, ha='center', va='top',color='black') 
    ax.text(scale0x+500, scale0y-70, '500m', fontsize=8, ha='center', va='top',color='black') 
    ax.text(scale0x+1000, scale0y-70, '1000m', fontsize=8, ha='center', va='top',color='black') 
    #draw dots with legend under scale
    legendx = scale0x
    legendy = scale0y - 500
    
    ax.plot(legendx, legendy, 'o', color='blue')
    ax.plot(legendx, legendy-200, 'o', color='black')
    ax.plot(legendx, legendy-400, 'o', color='orange')
    
    ax.text(legendx+70, legendy, ': Regulation nodes', fontsize=12, ha='left', va='center',color='black') 
    ax.text(legendx+70, legendy-200, ': Non Reg. Load only nodes', fontsize=12, ha='left', va='center',color='black') 
    ax.text(legendx+70, legendy-400, ': Non Reg. Load and DG nodes', fontsize=12, ha='left', va='center',color='black') 
    
    #data legend coordinates
    datax = right-20
    datay = lower-100
    vall  = 'top'
    hall  = 'right'
    
    #draw ct data    
    ax.text(datax, datay, 
            'Number of Non Reg. nodes: '+str(N-NPV)+'\n'+
            'Number of Reg. nodes: '+str(NPV)+'\n'+
            'Total Rated Load: '+str(Sload)+' MVA\n'+
            'Total installed DG: '+str(Sdg)+' MVA\n'+
            'Total length: '+str(length)+' km',
            fontsize=12, ha=hall, va=vall,color='black') 
    
    ax.set_frame_on(False)
    plt.tight_layout()
    # Saving the plot to an image file
    fig.savefig(plt_name+'.eps', format='eps')
    plt.show()
    fig.clf()
    return

#############################################################
def create_random_nodes(N, Dmin, Alength):
    #prefill vectors
    nodes = np.zeros((N, 2))  #x,y

    #run over the nodes
    for ii in range(N):
        searching = True
        while searching:
            x = Alength*random.random() 
            y = Alength*random.random()
            checking = True
            for jj in range(ii):
                d = np.sqrt((x-nodes[jj,0])**2 + (y-nodes[jj,1])**2)
                if d<Dmin:
                    checking = False
            if checking:
                searching = False
        nodes[ii,:] = [x,y]
    return nodes
#############################################################
def create_east_west_clustered_nodes(N, Dmin, Alength):
    #prefill vectors
    nodes = np.zeros((N, 2))  #x,y

    #run over the nodes
    for ii in range(N):
    
        searching = True
        while searching:
            x = Alength*random.random()/3 #west!
            if ii<(N/2):
                x = x + Alength*2/3 #east!
            y = Alength*random.random()
            checking = True
            for jj in range(ii):
                d = np.sqrt((x-nodes[jj,0])**2 + (y-nodes[jj,1])**2)
                if d<Dmin:
                    checking = False
            if checking:
                searching = False
        nodes[ii,:] = [x,y]
    return nodes
#############################################################
def create_picaron_nodes(N, Dmin, Alength):
    #prefill vectors
    nodes = np.zeros((N, 2))  #x,y

    rho0 = Alength/4
    rho1 = Alength/2
    std_dev = Alength/32

    #run over the nodes
    for ii in range(N):
        searching = True
        while searching:
            theta = random.random() * 2 * math.pi
            dist  = random.random() *(rho1-rho0) + rho0 #random.gauss(rho0, std_dev)
            
            x = Alength/2 +  dist*np.cos(theta)
            y = Alength/2 +  dist*np.sin(theta)
            
            checking = True
            for jj in range(ii):
                d = np.sqrt((x-nodes[jj,0])**2 + (y-nodes[jj,1])**2)
                if d<Dmin:
                    checking = False
            if checking:
                searching = False
        nodes[ii,:] = [x,y]
    return nodes

#############################################################
def intersected_lines(newii,newjj,lines,nodes):
    intersection = False
    Nlines = lines.shape[0]
    if newii!=0 or newjj!=0:
        xi  = nodes[newii,0]
        yi  = nodes[newii,1]
        xj  = nodes[newjj,0]
        yj  = nodes[newjj,1]
        
        mij = (yj-yi)/(xj-xi)
        cij = mij*xj - yj
                
        for gg in range(Nlines):
            aa = int(lines[gg,0])
            bb = int(lines[gg,1])
            if aa!=0 or bb!=0:
                xa  = nodes[aa,0]
                ya  = nodes[aa,1]
                xb  = nodes[bb,0]
                yb  = nodes[bb,1]
                
                mAB = (yb-ya)/(xb-xa)
                cAB = mAB*xb - yb
                
                if mij==mAB and cij==cAB:
                    intersection = True
                elif mAB!=mij:
                    #find intersection
                    xinter = -(cij-cAB)/(mAB-mij)
                    yinter = mij*xinter + cij
                    
                    cond1  = xinter<max(xi,xj)
                    cond2  = xinter>min(xi,xj)
                    
                    cond3  = xinter<max(xa,xb)
                    cond4  = xinter>min(xa,xb)
                    
                    condition = cond1 and cond2 and cond3 and cond4
                    
                    if condition :
                        intersection = True
    
    return intersection
#############################################################
def is_neighbor(ii,jj,lines):
    neighborhood = False
    if ii==jj:
        neighborhood = True
    else:
        for line in lines:
            aa = int(line[0])
            bb = int(line[1])
            if aa==ii and bb==jj:
                neighborhood = True
            if aa==jj and bb==ii:
                neighborhood = True
        
    return neighborhood
#############################################################
def is_near(ii,jj,lines,nodes):
    #check if proposed line between ii and jj has a small angle with existing lines
    near = False
    if ii==jj:
        near = True
    #check for angle between lines
    if not(near):
        for line in lines:
            aa = int(line[0])
            bb = int(line[1])
            cosmax = 0.7 # 10degrees =0.985; 45dregree = 0.7071
            if aa==ii and bb!=jj: #pivot in aa==ii
                xi = nodes[ii,0]
                yi = nodes[ii,1]
                xj = nodes[jj,0]
                yj = nodes[jj,1]
                xb = nodes[bb,0]
                yb = nodes[bb,1]
                
                #cosine theo
                a2 = (xi-xj)**2 + (yi-yj)**2 #ii-jj
                b2 = (xi-xb)**2 + (yi-yb)**2 #ii-bb
                c2 = (xj-xb)**2 + (yj-yb)**2 #jj-bb
                
                cosgam = (a2 + b2 - c2)/(2*np.sqrt(a2*b2)) #cos(gamma)
                
                if cosgam>cosmax: 
                    near = True
            elif aa==jj and bb!=ii: #pivot in aa==jj
                xi = nodes[ii,0]
                yi = nodes[ii,1]
                xj = nodes[jj,0]
                yj = nodes[jj,1]
                xb = nodes[bb,0]
                yb = nodes[bb,1]
                
                #cosine theo
                a2 = (xi-xj)**2 + (yi-yj)**2 #ii-jj
                b2 = (xj-xb)**2 + (yj-yb)**2 #jj-bb
                c2 = (xi-xb)**2 + (yi-yb)**2 #ii-bb
                
                cosgam = (a2 + b2 - c2)/(2*np.sqrt(a2*b2)) #cos(gamma)
                
                if cosgam>cosmax:
                    near = True
            elif bb==ii and aa!=jj: #pivot in bb==ii
                xi = nodes[ii,0]
                yi = nodes[ii,1]
                xj = nodes[jj,0]
                yj = nodes[jj,1]
                xa = nodes[aa,0]
                ya = nodes[aa,1]
                
                #cosine theo
                a2 = (xi-xj)**2 + (yi-yj)**2 #ii-jj
                b2 = (xi-xa)**2 + (yi-ya)**2 #ii-aa
                c2 = (xj-xa)**2 + (yj-ya)**2 #jj-aa
                
                cosgam = (a2 + b2 - c2)/(2*np.sqrt(a2*b2)) #cos(gamma)
                
                if cosgam>cosmax:
                    near = True
            elif bb==jj and aa!=ii: #pivot in bb==jj
                xi = nodes[ii,0]
                yi = nodes[ii,1]
                xj = nodes[jj,0]
                yj = nodes[jj,1]
                xa = nodes[aa,0]
                ya = nodes[aa,1]
                
                #cosine theo
                a2 = (xi-xj)**2 + (yi-yj)**2 #ii-jj
                b2 = (xj-xa)**2 + (yj-ya)**2 #jj-aa
                c2 = (xi-xa)**2 + (yi-ya)**2 #ii-aa
                
                cosgam = (a2 + b2 - c2)/(2*np.sqrt(a2*b2)) #cos(gamma)
                
                if cosgam>cosmax:
                    near = True
    return near
#############################################################
def create_random_lines(nodes, Nlines, dmax):
    lines     = np.zeros((Nlines, 3))  #norigin,ndestiny,distance
    N         = nodes.shape[0] #number of nodes
    
    #run over nodes to connect to their closests
    for ii in range(N):
        xi    = nodes[ii,0]
        yi    = nodes[ii,1]
        #find its closests node
        dmin  = float('inf')
        jjclosest = ii
        for jj in range(N):
            condition = not(is_neighbor(ii,jj,lines))
            condition = condition and not(intersected_lines(ii,jj,lines,nodes)) 
            condition = condition and not(is_near(ii,jj,lines,nodes))
            if condition:
                xj = nodes[jj,0]
                yj = nodes[jj,1]
                d  = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d<dmin and d<dmax:
                    jjclosest = jj
                    dmin      = d
        jj   = jjclosest
        xj   = nodes[jj,0]
        yj   = nodes[jj,1]
        dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
        
        if ii<Nlines:
            lines[ii,:] = [int(ii),int(jj),dist]
    
    #connect random nodes if possible
    for ll in range(Nlines):
        ii = int(lines[ll,0])
        jj = int(lines[ll,1])
                    
        if ii==jj:
            ii = random.randint(0, N-1)
            xi    = nodes[ii,0]
            yi    = nodes[ii,1]
            #find its closests node
            dmin  = float('inf')
            jjclosest = ii
            for jj in range(N):
                condition = not(is_neighbor(ii,jj,lines))
                condition = condition and not(intersected_lines(ii,jj,lines,nodes))
                condition = condition and not(is_near(ii,jj,lines,nodes))
                if condition:
                    xj = nodes[jj,0]
                    yj = nodes[jj,1]
                    d  = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if d<dmin and d<dmax:
                        jjclosest = jj
                        dmin      = d
            jj   = jjclosest
            xj   = nodes[jj,0]
            yj   = nodes[jj,1]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            
            lines[ll,:] = [int(ii),int(jj),dist]
     
    #split lines which are close to some node
    for ll in range(Nlines):
        ii = int(lines[ll,0])
        jj = int(lines[ll,1])
        xi = nodes[ii,0]
        yi = nodes[ii,1]
        xj = nodes[jj,0]
        yj = nodes[jj,1]
        c2 = (xj - xi)**2 + (yj - yi)**2
        #search for near nodes nn
        for nn in range(N):
            if nn!=ii and nn!=jj:
                xn = nodes[nn,0]
                yn = nodes[nn,1]
                
                a2 = (xi - xn)**2 + (yi - yn)**2
                b2 = (xj - xn)**2 + (yj - yn)**2
                
                dijn = np.sqrt( abs( b2  - ((b2+c2-a2)**2)/(4*c2)))
                
                #if distance is small: break line in two
                if dijn<75 and a2<c2 and b2<c2:
                    #use existing line
                    lines[ll,0] = ii
                    lines[ll,1] = nn
                    lines[ll,2] = np.sqrt(a2)
                    #look for another empty line
                    for llll in range(ll+1,Nlines):
                        if llll!=ll and lines[llll,2]==0:
                            lines[llll,0] = jj
                            lines[llll,1] = nn
                            lines[llll,2] = np.sqrt(b2)
                            break
                    break
    #origin node always smaller than destiny
    for ll in lines:
        if ll[0]>ll[1]:
            aux   = ll[0]
            ll[0] = ll[1]
            ll[1] = aux
    #eliminate repeated lines
    lines = np.unique(lines, axis=0)
    #remove lines that could not be connected
    lines = lines[lines[:,2]!=0]
    #sort lines accoding to nodes
    sorted_indices = np.lexsort((lines[:, 1], lines[:, 0]))
    lines = lines[sorted_indices]
            
    return lines
#############################################################
def select_reg_nodes(nodes, lines, NPV):
    N       = nodes.shape[0] #number of nodes
    PQnodes = []
    PVnodes = []
    
    #find nodes with the most neighbours
    Nneigbours = np.zeros((N, 1))
    for ll in lines:
        n1 = int(ll[0])
        n2 = int(ll[1])
        Nneigbours[n1] = Nneigbours[n1] + 1
        Nneigbours[n2] = Nneigbours[n2] + 1
    sorted_nodes = sorted(range(len(Nneigbours)), key=lambda i: Nneigbours[i], reverse=True)
    
    #add nodes to PV group
    for ii in sorted_nodes:
        if len(PVnodes)<NPV:
            if not(ii in PQnodes):#if it is not unregulated
                #add to regulation group
                PVnodes.append(ii)
                #add neighbors to unregulated group
                for ll in lines:
                    aa = int(ll[0])
                    bb = int(ll[1])
                    if ii == aa:
                        PQnodes.append(bb)
                    elif ii==bb:
                        PQnodes.append(aa)
    #clean repeated PQ nodes
    PQnodes = list(set(PQnodes))
    #add rest of nodes to PQ group
    for ii in range(N):
        if not(ii in PVnodes) and not(ii in PQnodes):
            PQnodes.append(ii)
    #N of PV
    NPV = int(len(PVnodes))
    #reorder nodes
    new_nodes = nodes*0
    nodes_new_position = np.zeros((N, 1))
    nn = 0
    for ii in PVnodes:
        new_nodes[nn,:] = nodes[ii,:]
        nodes_new_position[ii] = int(nn)
        nn = nn + 1
    for ii in PQnodes:
        new_nodes[nn,:] = nodes[ii,:]
        nodes_new_position[ii] = int(nn)
        nn = nn + 1
    #reorder lines
    for ll in lines:
        ll[0] = nodes_new_position[int(ll[0])]
        ll[1] = nodes_new_position[int(ll[1])]
    #origin node always smaller than destiny
    for ll in lines:
        if ll[0]>ll[1]:
            aux   = ll[0]
            ll[0] = ll[1]
            ll[1] = aux
    #sort lines accoding to nodes
    sorted_indices = np.lexsort((lines[:, 1], lines[:, 0]))
    lines = lines[sorted_indices]
    
    return new_nodes, lines, NPV

#############################################################
def select_external_reg_nodes(nodes, lines, NPV):
    N       = nodes.shape[0] #number of nodes
    PQnodes = []
    PVnodes = []
    
    #find center of mass
    mass_center = np.mean(nodes, axis=0)
    
    #order nodes from distance to center
    Distances = np.zeros((N, 1))
    ii = 0
    for nn in nodes:
        xii  = nn[0]
        yii  = nn[1]
        
        dii2 = (xii - mass_center[0])**2 + (yii - mass_center[1])**2
        Distances[ii] = dii2
        ii = ii + 1
                
    sorted_nodes = sorted(range(len(Distances)), key=lambda i: Distances[i], reverse=True)
    
    #add nodes to PV group
    for ii in sorted_nodes:
        if len(PVnodes)<NPV:
            if not(ii in PQnodes):#if it is not unregulated
                #add to regulation group
                PVnodes.append(ii)
                #add neighbors to unregulated group
                for ll in lines:
                    aa = int(ll[0])
                    bb = int(ll[1])
                    if ii == aa:
                        PQnodes.append(bb)
                    elif ii==bb:
                        PQnodes.append(aa)
                #add near nodes to unregulated group
                xii = nodes[ii,0]
                yii = nodes[ii,1]
                for jj in range(N):
                    if jj != ii:
                        xjj  = nodes[jj,0]
                        yjj  = nodes[jj,1]
                        dij2 = (xii-xjj)**2 + (yii-yjj)**2
                        if dij2<700**2: #closer than 700m
                            PQnodes.append(jj)
                    
    #clean repeated PQ nodes
    PQnodes = list(set(PQnodes))
    #add rest of nodes to PQ group
    for ii in range(N):
        if not(ii in PVnodes) and not(ii in PQnodes):
            PQnodes.append(ii)
    #N of PV
    NPV = int(len(PVnodes))
    #reorder PV nodes
    new_nodes = nodes*0
    nodes_new_position = np.zeros((N, 1))
    nn = 0
    for ii in PVnodes:
        new_nodes[nn,:] = nodes[ii,:]
        nodes_new_position[ii] = int(nn)
        nn = nn + 1
    
    #reorder PQ nodes from node 1
    Distances = np.zeros((len(PQnodes), 1))
    ii = 0
    for PQnn in PQnodes:
        xii  = nodes[PQnn,0] - new_nodes[0,0]
        yii  = nodes[PQnn,1] - new_nodes[0,1]
        
        dii2 = xii**2 + yii**2
        Distances[ii] = dii2
        ii = ii + 1
    
    sorted_nodes = sorted(range(len(Distances)), key=lambda i: Distances[i], reverse=False)
    
    for ii in sorted_nodes:
        new_nodes[nn,:] = nodes[PQnodes[ii],:]
        nodes_new_position[PQnodes[ii]] = int(nn)
        nn = nn + 1
    #reorder lines
    for ll in lines:
        ll[0] = nodes_new_position[int(ll[0])]
        ll[1] = nodes_new_position[int(ll[1])]
    #origin node always smaller than destiny
    for ll in lines:
        if ll[0]>ll[1]:
            aux   = ll[0]
            ll[0] = ll[1]
            ll[1] = aux
    #sort lines accoding to nodes
    sorted_indices = np.lexsort((lines[:, 1], lines[:, 0]))
    lines = lines[sorted_indices]
    
    return new_nodes, lines, NPV























