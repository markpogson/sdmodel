# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:47:14 2023

@author: mark.pogson@liverpool.ac.uk

this script contains functions used by social_diffusion.py

"""
from pyvis.network import Network
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpc
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd
import scipy
import numpy as np
import os
import string
import pickle
np.random.seed(0)

letters = string.ascii_lowercase

class file():
    def make_folder(*names):  # names is a folder name with optional subfolder names, which this function makes and returns the path of
        path = os.path.join(*names)
        if not os.path.isdir(path): os.makedirs(path) # n.b. if folder doesn't exist, new results will just be added to/replace existing ones
        return path
    def write_params(params, param_names, folder=''): # write parameters to a human-readable text file, summarising large lists as sets
        fold = file.make_folder(folder,'data')
        with open(os.path.join(fold,'params.txt'),'w') as f:
            for param,param_name in zip(params,param_names):
                f.write(param_name+' = '+str(param)+'\n')
        return
    def pickle_data(data,name,folder=''): # pickle data
        fold = file.make_folder(folder,'data',name)
        with open(os.path.join(fold,name+'.pkl'),'wb') as f:
            pickle.dump(data,f)
        return
    def to_csv(data,name,header='',folder=''): # write data to csv files
        fold = file.make_folder(folder,'data',name)
        for i,case in enumerate(data):
            if len(np.shape(data))==2: 
                fname = name+'_case_'+str(i)
                np.savetxt(os.path.join(fold,fname+'.csv'),case,fmt='%.2f',delimiter=',',header=header)
            else:
                for j,rep in enumerate(case):
                    fname = name+'_case_'+str(i)+'_rep_'+str(j)
                    np.savetxt(os.path.join(fold,fname+'.csv'),rep,fmt='%.2f',delimiter=',',header=header)
        return

class initialise():
    def get_params(n,nwpks, network_kind,consolidated,stratified,nwp,frac_c0,frac_c1,frac_e,beta_e,beta_f,k_e,k_f,r_e,r_f,af0,group_tf,aft,merge,shuffle,case=None,rep=None,folder=''):
        # unpack collected parameters
        nwp0,nwp1 = nwp
        group_tf0,group_tf1 = group_tf
        # regarding network_kind, consolidated and stratified parameters:
            # these are included as sensitivity parameters to allow easy comparison of their differing values, but they do not behave like the agent-level parameters
            # consolidated is used only inside this function, so does not need to be output
            # stratified and network_kind are not used in this function, but are needed outside it, so are output
        # create network parameter dictionary, n.b. all the other params in this function (except consolidated and stratified) refer to agents, but dealing with network at the same time
        if len(nwpks)==0 or nwp0==None: # i.e. no network parameters to use
            network_params = {}
        else:
            nwps = [nwp0] if nwp1==None else [nwp0,nwp1] # i.e. only keep defined parameters
            network_params = dict(zip(nwpks,nwps))
        # obtain number of each agent group from total n and group fractions
        frac_c = frac_c1 + frac_c0*frac_c1
        n_c0 = int(np.round(n*frac_c0*frac_c1)) # number of committed agents with strategy 0 (note frac_c0 is relative to frac_c1)
        n_c1 = int(np.round(n*frac_c1)) # " " strategy 1
        n_c = n_c0 + n_c1
        n_e = int(np.round(n*frac_e*(1-frac_c))) # number of explorers
        n_f = n - n_c - n_e # number of followers
        # parameter values for each agent, based on groups (ordered as committed, explorers, followers, with dummy values for committed)
        betas = np.array([0 if not merge else [beta_e,beta_f][i%2] for i in range(n_c)] + [beta_e for _ in range(n_e)] + [beta_f for _ in range(n_f)]) # rationality
        ks = np.array([0 if not merge else [k_e,k_f][i%2] for i in range(n_c)] + [k_e for _ in range(n_e)] + [k_f for _ in range(n_f)]) # trend-seeking
        rs = np.array([0 if not merge else [r_e,r_f][i%2] for i in range(n_c)] + [r_e for _ in range(n_e)] + [r_f for _ in range(n_f)]) # inertia
        bs = np.ones(n)-ks-rs # social coordination for each agent, so k+r+b = 1
        # initial strategies and committed status of agents
        active_ss = [0 if i%af0==0 else 1 for i in range(n-n_c)] # split in initial strategies with ratio af0 for strategy 0, randomly distributed across active agents
        ss = np.array([0 for _ in range(n_c0)] + [1 for _ in range(n_c1)] + active_ss, dtype=int) # initial strategies of agents
        cs = np.array([True for _ in range(n_c)] + [False for _ in range(n-n_c)]) # committed flag (can't rely on initial strategy since can be committed to 0 updated model)
        es = np.array([False for _ in range(n_c)] + [True for _ in range(n_e)] + [False for _ in range(n_f)])
        # time differences used in trend-seeking for each agent
        if consolidated: # i.e. align tfs with explorers/followers
            active_tfs = [group_tf0 for _ in range(n_e)] + [group_tf1 for _ in range(n_f)]
        else: # i.e. spread tfs evenly across all active agents
            active_tfs = [group_tf0 if i%aft==0 else group_tf1 for i in range(n-n_c)]
        if merge:
            afth = 2 if consolidated else min((aft,2)) # for simplicity, use 2 equal groups regardless, unless aft is 1, in which case match this
            committed_tfs = [group_tf0 if i%afth==0 else group_tf1 for i in range(n_c)]
        else:
            committed_tfs = [0 for _ in range(n_c)]
        tfs = np.array(committed_tfs+active_tfs, dtype=int)
        if case!=None:
            lrep = '_case_'+str(case)
            if not shuffle: rep = None # shuffle is the only cause of non-constant values in a case - just overwrite existing file if not shuffle, as no values changed
            if rep!=None: lrep+= '_rep_'+str(rep)
            agent_props = np.transpose(np.vstack((betas,ks,rs,bs,ss,cs,es,tfs)))
            fname = os.path.join(folder,'data','agent_props'+lrep+'.csv')
            header = 'betas,ks,rs,bs,ss,cs,es,tfs'
            np.savetxt(fname,agent_props,fmt='%.2f',delimiter=',',header=header)
        if shuffle: # shuffle agent positions so preferential attachment in the scale-free network no longer favours committed agents, which are listed first if not shuffled
            inds = np.random.permutation(range(n)) # randomised order for agents
            betas = betas[inds] # agent property values in randomised order
            ks = ks[inds] # note that the order is consisent across parameters, so agents retain their properties
            rs = rs[inds] # also note that if merge=True, property values have already been assigned to committed agents so they appear like active agents for homophily
            bs = bs[inds]
            ss = ss[inds]
            cs = cs[inds]
            es = es[inds]
            tfs = tfs[inds]
        return stratified,network_kind,network_params,betas,ks,rs,bs,ss,cs,es,tfs
    
class network(): # see https://networkx.org/documentation/stable/reference/generators.html for numerous options, and Sikder (2020) for common ones for social networks
    def scalefree_pdf(x,g=2.1,a_0=0.01):
        return (1-g)/(1-a_0**(1-g))*x**(-g)
    def scalefree_dist(g=2.1,a_0=0.01): # generate random number from dist F(x) ~ x^(-g) in range [a_0,1]; use this to give scale-free network
        if g==1:
            return('g must not equal 1 (divide by zero error)')
        scalefree_pdf_max = network.scalefree_pdf(a_0,g,a_0) # max of scalefree_pdf in range [a_0,g]
        for _ in range(999999): # arbitrary limit to number of attempts tp obtain suitable random number
            x = np.random.uniform(a_0,1) # i.e. maximum possible message rate is 1, minimum is a_0
            if np.random.rand() < network.scalefree_pdf(x,g,a_0)/(scalefree_pdf_max+1): # i.e. use x if random number falls under rescaled scalefree_pdf curve (rescaling is arbitrary so long as max<1; smaller curve will result in taking more iterations to get result, but distribution unaffected)
                return x
    def check_symmetric(a, tol=1e-8): # see https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
        return np.all(np.abs(a-a.T) < tol)
    def get_adjacencies(n,kind='complete',network_params={},hs=[],seed=0): # hs are any values (e.g. bs,tfs) which can be used to provide homophily in 'baumann_scalefree'
        network_params = network_params.copy()
        if kind=='complete': # fully-connected network with 2-way connections
            A = np.ones((n,n)) - np.eye(n) # matrix of ones with zeros on diagonal, i.e. excluding self-connection
        elif kind=='regular': # each node has same degree, with undirected connections
            if not network_params:
                network_params = dict(d=0.05)
            network_params['d'] = int(n*network_params['d'])
            if network_params['d']<1: network_params['d'] = 1
            if (network_params['d']*n)%2!=0:    
                network_params['d']+=1 # forcing d*n to be even, as required by the graph function
            G = nx.random_regular_graph(n=n, **network_params, seed=seed) # d is degree of each node, n is number of nodes
        elif kind=='small world': # undirected connections with small-world network, i.e. most nodes unconnected to each other, but most neighbours have other neighbours in common, and degrees of separation between nodes mostly small
            if not network_params:
                network_params = dict(k=0.05,p=0.1)
            network_params['k'] = int(n*network_params['k'])
            if network_params['k']<1: network_params['k'] = 1
            G = nx.newman_watts_strogatz_graph(n=n, **network_params, seed=seed) # small world graph where k is number of nearest neighbours and p is probability of adding a new edge for each edge
        elif kind=='scale free': # undirected scalefree network, using barabasi-albert model
            if not network_params:
                network_params = dict(m=0.05)
            network_params['m'] = int(n*network_params['m'])
            if network_params['m']<1: network_params['m'] = 1
            G = nx.barabasi_albert_graph(n=n, **network_params, seed=seed) # m (in network_params dictionary) is number of edges from each new node
        elif kind=='directed' or kind=='homophilous': # directed scale-free network based on Baumann et al. (2020) with directed connections, reworked here to be static (and homophilous if hs given)
            if not network_params:
                network_params = dict(g=2,a_0=5/n)
            agents = np.arange(n)
            # generate number of followers for each agent
            fracs = [network.scalefree_dist(**network_params) for _ in agents] # fraction of population that are followers for each agent, with minimum value 1/n to ensure no isolated agents
            nlinks = [int(np.round(np.clip(n*frac,0,n))) for frac in fracs] # number of followers for each agent (ensuring multiplied fraction gives between 1 and n, but this is guaranteed by a0 and scalefree dist output anyway)
            followers = []
            # assign specific followers, either at random or based on homophily of bs (i.e. bs are ordered by absolute difference from bs[i])
            for i,nlink in enumerate(nlinks):
                if kind=='directed':
                    followers.append(np.random.permutation(agents[agents!=i])[:nlink]) # i.e. pick nlink random agents for outward connections
                else: # homophilous
                    hh = abs(hs-hs[i])
                    hh[i] = max(hh)+9 # easy way to exclude self-messaging, i.e. artificially make agent i the least homophilous agent to itself
                    ran = np.random.random(hh.size) # effectively gives random permuation if ordered by value
                    sor = np.lexsort((ran,hh)) # sort by hh, then ran for equal hh values, see https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
                    followers.append(sor[:nlink]) # i.e. pick the nlink agents with the closest b value to b[i], randomised when there's nothing to choose between them
            # put agent connections into adjacency matrix
            A = np.zeros((n,n))
            for i,fol in enumerate(followers):
                A[i,fol] = 1 # making adjency matrix reflect followers
        else:
            print('error: network name not recognised in get_adjacencies()')
            return
        if kind!='complete' and kind!='directed' and kind!='homophilous': # need to convert from networkx graph to adjacency matrix as numpy array
            A = nx.to_numpy_array(G)
        #print(network.check_symmetric(A))
        return A # agent i has followers j, where i is row and j is column of A
    def get_trusts(A,stratified=True):
        if not stratified: # use A as the trust matrix, but this will still need to be normalised by following counts for consistency
            T = A # i.e. no weighted difference in agents' ability to influence
        else: # use follower counts to construct unnormalised adjacency matrix
            n = len(A)
            follower_counts = np.sum(A,axis=1) # status of agents
            follower_counts = np.repeat( # need to transpose/repeat/reshape to multiply with A
                np.transpose(follower_counts),n).reshape([n,n])
            T = follower_counts*A # unnormalised trust between agents (normalisation is specific to each follower agent, based on the sum of the total trust it puts in other agents) 
        # normalise trust matrix (so each column sums to 1)
        following_counts = np.sum(T,axis=0) # n.b. with certain networks some following counts may be zero, which will give an invalid value warning in divide, but safe to ignore as agent will be unaffected by any other agents anyway
        with np.errstate(divide='ignore', invalid='ignore'): # see above line for explanation of why these errors don't matter
            T /= following_counts # this gives the adjacency values normalised by the sum in each colummn (i.e. each follower has a total trust of 1 spread across agents it is following)
        T = np.nan_to_num(T) # making nans 0 (default arguments do this) 
        return T # this is the only place the trust matrix is referred to as T in the code - it's A everywhere else
    def get_graph(A,multi=False):
        if multi:
            gx = nx.MultiDiGraph() # networkx graph with scope for multiple directed edges between nodes
        else:
            gx = nx.DiGraph() # note this is sufficient for the networks being considered, which only have single connections between node pairs, which may be bidirectional, but are not multiple
        n = len(A) # A always square matrix
        for i in range(n):
            for j in range(n):
                if A[i,j]!=None and A[i,j]!=0:
                    gx.add_edge(str(i),str(j))
        return gx

class simulate():
    def diffuse(A,betas,ks,rs,bs,ss,cs,tfs,tmax=500):    
        '''
        inputs:
            A, adjacency matrix (which may have been normalised to be a trust matrix if stratified=True, but keep same notation)
            betas, rationality of each agent
            ks, inertia " "
            rs, trend-seeking " "
            bs, social coordination " "
            ss, initial strategy " "
            cs, flag for committed agents (either for strategy 1 or 0)
            tfs, time period each agent calculates rate of change over
        local variables:
            xs, strategy of agents (i.e. 0 or 1); vector length n (the time series of these will be stored in xall, a matrix of shape (tmax,n))
            xold, previous strategy of agents
            toffs, random offset for start of timescale, up to timescale itself - this avoids misleading coordination between agents
        '''
        def get_tpast(t,tf,toff): # current time t, agent timescale tf, agent time offset toff
            tpast = np.clip(t-tf+toff,0,t) # i.e. subtract timescale from current time, but add on offset so agents aren't coordinated
            return tpast
        n = len(A)
        xold = np.array([s for s in ss]) # previous x for each agent (copy.copy and copy.deepcopy are unreliable to avoid entries being pointers)
        xs = np.array([x for x in xold]) # current x " "
        xall = np.zeros([tmax,n])
        xall[0,:] = xold
        xall[1,:] = xs
        toffs = [0 if tf<2 else np.random.randint(tf) for tf in tfs] # i.e. each offset is a random integer up to (but not including) the agent timescale
        for t in range(1,tmax): # start from t=1 so as not to overwrite initial xs
            xhats = simulate.get_xhats(A,xs,xold) # Eq. 3, i.e. trend-seeking values
            E0 = simulate.E(A,1-xs,ks,bs,rs,1-xhats) # Eq. 2b in Ye et al., i.e. expected payoff of strategy 0 for each agent
            E1 = simulate.E(A,xs,ks,bs,rs,xhats) # Eq. 2a " " strategy 1 " "
            p1s = np.exp(betas*E1)/(np.exp(betas*E1)+np.exp(betas*E0)) # Eq. 1 " ", i.e. probability of to transition to 1 (using E0 in the denominator would give the opposing probability, and these would sum to 1)
            xs = np.zeros(n) # this effectively deals with the reverse transtion, with committed and transitioned agents dealt with below
            rdn = np.random.rand(n) # random number in range [0,1) for each agent
            xs[rdn<p1s] = 1 # i.e. setting strategy as 1 if random number is less than calculated probability
            xs[cs==True] = ss[cs==True] # i.e. committed agent strategies made to remain as 1
            xall[t,:] = xs
            xold = np.array([xall[get_tpast(t,tfs[i],toffs[i]),i] for i in range(n)])
        return xall
    def E(A,xs,ks,bs,rs,xhats): # Eq. 2a/b in Ye et al. (Eq. 2b obtained by inputting 1-xs and 1-xhats)
        n = len(xs)
        social_coordination = np.zeros(n) # this will be the social coordination term for each agent
        for i in range(n): # iteration makes this clearer than before the adjacency matrix was present
            social_coordination[i] = bs[i]*sum(A[:,i]*xs) # A[:,i] is the normalised trust agent i gives each other agent j (note the axis being used in the array), which is multiplied element-wise by x values, and summed before being multiple by the social coordination parameter for i
        inertia = ks*xs
        trend_seeking = rs*xhats
        return social_coordination + inertia + trend_seeking # always in [0,1] due to input values also being in [0,1]
    def get_xhats(A,xs,xold): # could do away with A but keeping it for testing of complete network
        n = len(xs)
        xhats = np.zeros(n)
        for i in range(n):
            xhats[i] = 0.5*(1+(sum(A[:,i]*(xs-xold)))) #  trust and iteration simplify this calculation compared to before the adjacency matrix
        return xhats
    
class analyse():
    def get_flow(xal,A): # xal is xall up to current time step, A is adjacency matrix; returns number of agents which have received influence from an agent which has strategy 1
        n,n = np.shape(A)
        received = np.zeros(n) # this will contain 1 for agents which have received a signal of strategy 1
        for i in range(n):
            senders = A[:,i].astype(bool) # agents sending to i
            if np.sum(xal[:,senders])>0:
                received[i] = 1
        return received # i.e. whether signal has been received for each agent - will be filtered outside function
    def get_metrics(xall,A_,cs,es=[],period='pretime',frac_takeoff=0.2,frac_diffusion=0.9,frac_c0=0.,frac_c1=0.25):
        '''
        inputs:
            xall, matrix of strategy at each time point (rows) for each agent (columns)
            cs, flag for committed agents (either for strategy 1 or 0)
            period, time period to calculate swtiching count over, either 'pre','post','inter' or 'full' - these are explained in the function code
            frac_takeoff, fraction of active agents changed from starting strategy required for take-off
            frac_diffusion, " " full diffusion
        local variables:
            tmax, number of time steps to iterate over
            xs, strategy of agents (i.e. 0 or 1); vector length n (the time series of these will be stored in xall, a matrix of shape (tmax,n))
            ys, switching count (will be converted to rate at end of function) of agents; vector length n
            zs, fraction of non-committed agents with strategy 1; vector length t
            Tk, take-off time
            Td, diffusion time
        information flow:
            note that I've replicated the above but for information flow, where I've assumed all active agents are initialised with strategy 0 since this is always how the model is used, but could allow either strategy by using the same approach as for social diffusion
        '''
        # initialise values
        A = analyse.denormalise(A_)
        tmax,n = np.shape(xall)
        ys = np.zeros(n) # will be a running sum for number of switches in strategy for each agent
        zs = [] # will be time-series cumulative total of non-committed agents with strategy 1
        taken_off = False
        diffused = False
        taken_offI = False # information equivalent of social diffusion cardinal times
        diffusedI = False
        n_active = sum(~cs)
        is_00 = ~cs & (xall[0,:]==0) # flag for active agents initialised with strategy 0
        is_10 = ~cs & (xall[0,:]==1) # " " 1
        t_start = 0 if period=='pre' or period=='full' else tmax # if period='post' or period='inter', this will modified in the loop
        t_stop = tmax # if period='pre' or period='inter', this will modified in the loop
        Tk = tmax; Td = tmax # these will be replaced by the actual times if/when they're reached
        TkI = tmax; TdI = tmax # these will be replaced by the actual times if/when they're reached
        zis = [] # this will be time-series cumulative total of non-committed agents which have been messaged by strategy 1
        if len(es)>0:
            ys_e = np.zeros(n)
            zs_e = []
            zis_e = []
            ys_f = np.zeros(n)
            zs_f = []
            zis_f = []
            taken_off_e = False
            diffused_e = False
            taken_off_eI = False
            diffused_eI = False
            n_active_e = sum(~cs&es)
            taken_off_f = False
            diffused_f = False
            taken_off_fI = False
            diffused_fI = False
            n_active_f = sum(~cs&~es)
            is_00_e = ~cs & es & (xall[0,:]==0) # flag for explorer agents initialised with strategy 0
            is_10_e = ~cs & es & (xall[0,:]==1) # " " 1
            is_00_f = ~cs & ~es & (xall[0,:]==0) # flag for laggard initialised with strategy 0
            is_10_f = ~cs & ~es & (xall[0,:]==1) # " " 1
            Tk_e = tmax; Td_e = tmax; Tk_f = tmax; Td_f = tmax
            t_start_e = t_start; t_stop_e = t_stop; t_start_f = t_start; t_stop_f = t_stop;
            Tk_eI = tmax; Td_eI = tmax; Tk_fI = tmax; Td_fI = tmax
            t_start_eI = t_start; t_stop_eI = t_stop; t_start_fI = t_start; t_stop_fI = t_stop; 
        # iterate over xall to obtain metrics
        for t in range(tmax):
            xs = xall[t,:]
            # switching rate metric
            change = xs - xall[t-1,:] # note that t-1 is correct here, unlike xhats which needs the time difference parameter for the agent
            if t>t_start and t<=t_stop: # note the first time point is excluded from time range as there's no time 0-1
                ys += abs(change) # running sum for Eq. 4d in Ye et al., i.e. cumulative number of switches for each agent within specified time range
            # takeoff and diffusion time metrics
            n_switched = sum(xs[is_00]==1)+sum(xs[is_10]==0) # i.e. number of active agents at current time step that have switched from their original strategy
            frac_switched = n_switched/n_active
            if not taken_off and frac_switched>=frac_takeoff:
                Tk = t # take-off time
                taken_off = True
                if period=='intertime': # intertime is between take-off and diffusion time
                    t_start = t-1 # this means ys will start counting at the next time step
            if not diffused and frac_switched>=frac_diffusion:
                Td = t # diffusion time
                diffused = True
                if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                    t_stop = t
                elif period=='posttime':
                    t_start = t-1
            # diffusion time-series metric
            zs.append(sum(xs[~cs])/n_active) # i.e. number of active agents at current time step that have strategy 1
            # information diffusion time-series metric, i.e. number of agents that have at some point been exposed to strategy 1
            exposed = analyse.get_flow(xall[:t],A)
            frac_exposed = sum(exposed[~cs])/n_active
            zis.append(frac_exposed) # i.e. fraction of active agents that have received influence from an agent with strategy 1 at any time up to and including the present time step
            # information takeoff and diffusion time metrics
            if not taken_offI and frac_exposed>=frac_takeoff:
                TkI = t # take-off time
                taken_offI = True
                if period=='intertime': # intertime is between take-off and diffusion time
                    t_start = t-1 # this means ys will start counting at the next time step
            if not diffusedI and frac_exposed>=frac_diffusion:
                TdI = t # diffusion time
                diffusedI = True
                if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                    t_stop = t
                elif period=='posttime':
                    t_start = t-1
            if len(es)>0:
                # explorer and laggard switching rates
                if t>t_start_e and t<=t_stop_e: # note the first time point is excluded from time range as there's no time 0-1
                    ys_e += abs(change) # running sum for Eq. 4d in Ye et al., i.e. cumulative number of switches for each agent within specified time range
                if t>t_start_f and t<=t_stop_f: # note the first time point is excluded from time range as there's no time 0-1
                    ys_f += abs(change) # running sum for Eq. 4d in Ye et al., i.e. cumulative number of switches for each agent within specified time range
                # explorer and laggard time-series metric for diffusion and information
                zs_e.append(sum(xs[~cs&es])/n_active_e) # i.e. fraction of active agents at current time step that have strategy 1
                frac_exposed_e = sum(exposed[~cs&es])/n_active_e
                zis_e.append(frac_exposed_e)
                zs_f.append(sum(xs[~cs&~es])/n_active_f) # i.e. number of active agents at current time step that have strategy 1
                frac_exposed_f = sum(exposed[~cs&~es])/n_active_f
                zis_f.append(frac_exposed_f)
                # explorer take-off and diffusion time metrics
                n_switched_e = sum(xs[es&is_00_e]==1)+sum(xs[es&is_10_e]==0) # i.e. number of active agents at current time step that have switched from their original strategy
                frac_switched_e = n_switched_e/n_active_e
                n_switched_f = sum(xs[~es&is_00_f]==1)+sum(xs[~es&is_10_f]==0) # i.e. number of active agents at current time step that have switched from their original strategy
                frac_switched_f = n_switched_f/n_active_f
                if not taken_off_e and frac_switched_e>=frac_takeoff:
                    Tk_e = t # take-off time
                    taken_off_e = True
                    if period=='intertime': # intertime is between take-off and diffusion time
                        t_start_e = t-1 # this means ys will start counting at the next time step
                if not diffused_e and frac_switched_e>=frac_diffusion:
                    Td_e = t # diffusion time
                    diffused_e = True
                    if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                        t_stop_e = t
                    elif period=='posttime':
                        t_start_e = t-1
                # laggard take-off and diffusion time metrics
                n_switched_f = sum(xs[es&is_00_f]==1)+sum(xs[es&is_10_f]==0) # i.e. number of active agents at current time step that have switched from their original strategy
                frac_switched_f = n_switched_f/n_active_f
                n_switched_f = sum(xs[~es&is_00_f]==1)+sum(xs[~es&is_10_f]==0) # i.e. number of active agents at current time step that have switched from their original strategy
                frac_switched_f = n_switched_f/n_active_f
                if not taken_off_f and frac_switched_f>=frac_takeoff:
                    Tk_f = t # take-off time
                    taken_off_f = True
                    if period=='intertime': # intertime is between take-off and diffusion time
                        t_start_f = t-1 # this means ys will start counting at the next time step
                if not diffused_f and frac_switched_f>=frac_diffusion:
                    Td_f = t # diffusion time
                    diffused_f = True
                    if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                        t_stop_f = t
                    elif period=='posttime':
                        t_start_f = t-1
                # information flow explorer takeoff and diffusion time metrics
                if not taken_off_eI and frac_exposed_e>=frac_takeoff:
                    Tk_eI = t # take-off time
                    taken_off_eI = True
                    if period=='intertime': # intertime is between take-off and diffusion time
                        t_start_eI = t-1 # this means ys will start counting at the next time step
                if not diffused_eI and frac_exposed_e>=frac_diffusion:
                    Td_eI = t # diffusion time
                    diffused_eI = True
                    if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                        t_stop_eI = t
                    elif period=='posttime':
                        t_start_eI = t-1
                # information flow laggard take-off and diffusion time metrics
                if not taken_off_fI and frac_exposed_f>=frac_takeoff:
                    Tk_fI = t # take-off time
                    taken_off_fI = True
                    if period=='intertime': # intertime is between take-off and diffusion time
                        t_start_fI = t-1 # this means ys will start counting at the next time step
                if not diffused_fI and frac_exposed_f>=frac_diffusion:
                    Td_fI = t # diffusion time
                    diffused_fI = True
                    if period=='pretime' or period=='intertime': # pretime is pre-diffusion time (there's no option for pre-take-off time)
                        t_stop_fI = t
                    elif period=='posttime':
                        t_start_fI = t-1
        # rescale switching count to obtain switching
        ys = ys[~cs]/(t_stop-t_start) # Eq. 4d in Ye et al., applied to non-committed agents (not using -1, which appears in Ye et al. to sum up and excluding to diffusion time, but this is already excluded in the above loop)
        if len(es)>0:
            ys_e = ys_e[~cs&es]/(t_stop_e-t_start_e) # Eq. 4d in Ye et al., applied to non-committed agents (not using -1, which appears in Ye et al. to sum up and excluding to diffusion time, but this is already excluded in the above loop)
            ys_f = ys_f[~cs&~es]/(t_stop_f-t_start_f) # Eq. 4d in Ye et al., applied to non-committed agents (not using -1, which appears in Ye et al. to sum up and excluding to diffusion time, but this is already excluded in the above loop)
            # put cardinal times into lists to output
            Tk = [Tk,Tk_e,Tk_f] # note this is [all,explorers,laggards]
            Td = [Td,Td_e,Td_f]
            ys = [ys,ys_e,ys_f]
            #zs = [zs,zs_e,zs_f] # these are commented out as never use the breakdown of explorer and laggard dynamics, and would need to update plot_timeseries to make sense of this
            #zis = [zis,zis_e,zis_f]
            TkI = [TkI,Tk_eI,Tk_fI]
            TdI = [TdI,Td_eI,Td_fI]
        return Tk,Td,ys,np.array(zs),TkI,TdI,np.array(zis)
    def denormalise(A_):
        A = A_.copy()
        A[A!=0] = 1 # convert from trust to adjacency matrix (as want actual in- and out- degrees)
        return A
    def get_degrees(A_): # get indegree ('follower count') and outdegree ('following count') for each agent
        A = analyse.denormalise(A_)
        ins = np.sum(A,axis=0) # ins are follower counts, outs are following counts
        outs = np.sum(A,axis=1) # note that committed agents are included, which will bias outs (i.e. 0 for all committed agents)
        degs = ins + outs
        return degs
    def get_sps(A_,dev=False): # find shortest path length between every possible pair of nodes
        A = analyse.denormalise(A_)
        gx = network.get_graph(A)
        sps = nx.shortest_path_length(gx) # this is a generator object giving dictionaries for path distances for each node to each other node
        if dev: # return intermediate result for convenience during development
            return sps
        lsps = list(sps) # node indices
        lengths = [] # note the directions are accounted for, i.e. it is sent length
        for node in lsps:
            ls = list(node[1].values()) # node[0] is node the id, node[1] is the dictionary of path lengths
            ls = ls[1:] if len(ls)!=1 else [0] # ignore 1st entry which is distance to self (0 by definition), unless node is disconnected, in which case give 0 as shortest length
            lengths+=ls # lengths is a flat list of distances between all nodes
        return lengths
    def get_edges(A_,cs,es): # get count of edges between all groups
        A = analyse.denormalise(A_) # do with and without trusts (A_ is with trusts)
        # use adjacency/trust matrix to get relevant sums
        comexp = np.sum(A[cs][:,es]) # committed->explorer adjacency count
        comexp_ = np.sum(A_[cs][:,es]) # " " trust sums
        comlag = np.sum(A[cs][:,~cs&~es]) # note that the slicing can't be done with a comma, the rows are selected first, then the columns from these rows - this allows any combination of the two
        comlag_ = np.sum(A_[cs][:,~cs&~es])
        expexp = np.sum(A[es][:,es]) # explorer->explorer adjacency count
        expexp_ = np.sum(A_[es][:,es]) # " " trust sums
        explag = np.sum(A[es][:,~cs&~es])
        explag_ = np.sum(A_[es][:,~cs&~es])
        lagexp = np.sum(A[~cs&~es][:,es]) # laggard->explorer adjacency count
        lagexp_ = np.sum(A_[~cs&~es][:,es]) # " " trust sums
        laglag = np.sum(A[~cs&~es][:,~cs&~es])
        laglag_ = np.sum(A_[~cs&~es][:,~cs&~es])
        # put these into lists
        edgecounts = [comexp,comlag,expexp,explag,lagexp,laglag] # directed adjacency counts for committed->explorer, committed->lagard, explorer->explorer, explorer->laggard, lagard->explorer and laggard->laggard
        edgecounts_ = [comexp_,comlag_,expexp_,explag_,lagexp_,laglag_] # directed trust sums " "
        return [edgecounts,edgecounts_]
    def get_clusters(A_):
        A = analyse.denormalise(A_)
        gx = network.get_graph(A)
        clusters = nx.clustering(gx, nodes=None, weight=None) # dictionary of the clustering coefficient for each node key
        clusters = list(clusters.values())
        return clusters
    def get_flat(lis,ind): # get index(es) ind from list of reps in cases
        if isinstance(ind,int):
            return [[rep[ind] for rep in case] for case in lis]
        else:
            return [[[rep[which] for rep in case] for which in ind] for case in lis]
    
class plot():
    def add_spl(ax,spl): # add label spl inside brackets and in bold to top left of axes ax
        if spl: # i.e. only add spl is text exists, otherwise just get empty brackets
            xl = ax.get_xlim(); yl = ax.get_ylim()
            ax.text(-0.1, 1.1,'('+spl+')', weight='bold', transform=ax.transAxes) # transformed to use figure units, not plot units to specify x,y
        return ax
    def savefig(fname,dpi=200):
        if fname[-4]=='.png': # i.e. only want png file
            fexts = [''] # no need to add any extension
        else:
            fexts = ['.png','.svg']
        for fext in fexts:
            plt.savefig(fname+fext, dpi=dpi, bbox_inches='tight')
        return        
    def get_imed(Ts):
        imed = np.argsort(Ts)[len(Ts)//2] # index of median value (or first of 2 middle values) using floor division
        return imed
    def make_name(label):
        name = label.replace('$','').replace('\\','').replace(' ','')
        return name
    def get_base_colours():
        colours = list(mpc.BASE_COLORS.values()) # only used for non-numeric parameters to be consistent with other plot colours
        cyan = colours[3]; black = colours[6]
        colours[3] = black; colours[6] = cyan
        return colours
    def try_rotation(ax):
        ticklabs = ax.get_xticklabels()
        if any([len(tl.get_text())>6 for tl in ticklabs]) and not any(['$' in tl.get_text() for tl in ticklabs]): # $ indicates shorter label expected
            plt.xticks(rotation=40,ha='right')
        return ax
    def set_log(ax,vals):
        ax.set_yscale('log')
        ax.grid(which='minor',lw=0.2,axis='y')
        if min(vals)<=1: # do this after log scale
            ax.set_ylim((0.9,None)) # avoid issues with 0 and log scale - they're handled fine, but clearer to cut off slightly below 1
        return ax
    def make_legend(ax):
        ax.legend(bbox_to_anchor=(1,1),loc='upper left')
        return ax
    def plot_timeseries(zss_,Ts_,labels='',ts_pt=200,case_label='Case',heat=False,ythickness=2,xthickness=3,frac_takeoff=0.2,frac_diffusion=0.9,ylabel='Social Diffusion (%)',declutter=False,spl='',folder=''): # plot time series of z values for each trial, highlighting the trial which has median Td value
        colours = plot.get_base_colours()
        fig = plt.figure(figsize=[6,4]); fig.clf(); ax = fig.add_subplot()
        sh = np.shape(zss_)
        if len(sh)>3:
            ncase,nrep,_,nt = sh
        else:
            ncase,nrep,nt = sh
        if ts_pt==None or ts_pt>nt: ts_pt = nt
        if heat: # create array to store values for heatmap
            arr = np.zeros((ncase*nrep,ts_pt,4)) # rgba values
            def resc(v): # rescale values to make colour map easier to distinguish around take-off (log would need too much faffing and explaining)
                if v<frac_takeoff:
                    out = 0.2*v # i.e. make even fainter anything before take-off
                elif v<frac_diffusion:
                    out = 0.6*v # i.e. step change at takeoff, and linear up to diffusion time
                else:
                    out = v # i.e. second step change at diffusion time
                return out
        else:
            arr = []
        for l,(zss,Ts) in enumerate(zip(zss_,Ts_)): # i.e. each experiment
            if heat: # put reps in order to use in heatmap
                args = np.argsort(Ts)
                zss = np.array(zss.copy())[args]
            else: # get median index to highlight in lineplot
                imed = plot.get_imed(Ts) # i.e. median rep for trial
            for i,zs_ in enumerate(zss): # i.e. each rep
                zs = zs_.copy()[:ts_pt]
                if heat:
                    ind1 = (l+1)*nrep
                    for j in range(ts_pt): # assign colour based on case, and alpha based on z value for each time point
                        arr[ind1-i-1,j,:] = list(colours[l])+[resc(zs[j])]
                else:
                    lw = 1.2 if i==imed else 0.1
                    colour = colours[l]
                    label = ''
                    if labels!='' and i==imed: label = labels[l]
                    if i==imed or not declutter:
                        ax.plot(range(len(zs)),100*np.array(zs),c=colour,lw=lw,label=label) # i.e. plot fraction of non-committed agents with strategy 1 at each time step
        ax.set_xlabel('Time $t$')
        if heat:
            # show heatmap
            ax.imshow(arr,extent=[0,ts_pt,0,ncase*nrep],aspect='auto')
            # reverse y-axis so ascends from bottom up
            ax.invert_yaxis()
            # set ytick positions and labels
            ct = nrep # y-index thickness for case
            tt = nrep*ncase # y-index total thickness
            yticks = np.linspace((ct-1)/2,tt-(ct+1)/2,ncase) # -1 and +1 due to direction from start/end
            labs = labels[::-1] # reverse order of labels to be ascending on y-axis
            labs_ = [];param=''
            for lab in labs:
                if '=' in lab:
                    param,v = lab.split('=') # i.e. pick out specific value v to use as ticklabel
                    v = v.strip()
                    labs_.append(v)
                else:
                    labs_.append(lab)
            ax.set_yticks(yticks,labels=labs_) # ticks will be in middle of cases, i.e. aligned with medians
            ax.set_ylabel(param)
            # also deal with xticks
            xticks = np.linspace(0,ts_pt,9)-0.5 # -0.5 so ticks start from 0, not middle of first value
            xticklabels = np.linspace(0,ts_pt,9)
            ax.set_xticks(xticks,labels=xticklabels)
            # don't label yaxis, as yticklabels explain themselves
            #ax.set_ylabel(case_label)
            # draw vertical colour bar on right of plot
            ninc = 100 # number of vertical increments (leave as 10, as larger values stretch colour bar and are unnecessary anyway)
            yoff = 0.05 # vertical offset from 0
            xoff = 0.05 # horizontal offset from right of plot
            x0 = 1+xoff
            y0 = yoff
            w = 0.1 # width of colour bar in figure units
            h = 1-2*yoff # i.e. centred vertically
            yoff -= h
            cax = ax.inset_axes([x0, y0, w, h])
            cax.yaxis.set_label_position('right')
            cax.yaxis.tick_right()
            cax.xaxis.set_visible(False)
            cax.set_ylabel(ylabel)
            cbar = np.zeros((ninc,ncase,4))
            for row in range(ninc):
                for col in range(ncase):
                    cbar[row,col,:] = list(colours[col])+[resc(row/(ninc-1))] # -1 for closed interval
            cax.imshow(cbar,aspect='auto')
            cax.invert_yaxis() # gives upward scale
            cax.set_yticks(np.linspace(0,ninc,6)-.5,labels=np.linspace(0,100,6,dtype=int)) # -0.5 to start from 0 in terms of yticks, not indices
        else:
            ax.set_ylabel(ylabel)
            ax.set_ylim(0,100)
            if labels!='': plot.make_legend(ax)
        pref = 'heatmap' if heat else 'timeseries'
        name = plot.make_name(pref+'_tlim_'+str(ts_pt)+'_'+ylabel)
        plot.add_spl(ax,spl)
        plot.savefig(os.path.join(folder,name))
        plt.close(fig)
        return arr
    def plot_dist(yss_,Ts_,param_label,labels=[],xlabel='Switching Rate',boxplot=False,scatterplot=False,log=True,spl='',folder=''): # plot histogram of y values for the trial with the median Ts value (Ts is takeoff or diffusion time or another similar measure, yss_ contains yss for each trial, which contains ys for each rep, which contains y for each agent); y is the switching rate of each agent
        colours = plot.get_base_colours()
        if not boxplot and not scatterplot:
            flat_arr = np.array(yss_).ravel() # find sensible consistent bins and axis range across all possible reps
            xlim = (flat_arr.min(),flat_arr.max())
            _,bins = np.histogram(flat_arr) # note that this is for all cases lumped together, but actual plot(s) will only be for median of each sensitivity case
        # plot histogram of median rep for each sensitivity case
        if boxplot or scatterplot:
            fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
            if boxplot:
                k = -1 # index for parameter values
            else:
                poss = [] # store of scatterplot x-positions
                valss = [] # " " y " "
        for l,(yss,Ts) in enumerate(zip(yss_,Ts_)): # subscipt used to distinguish Ts_ of multiple trials from Ts of multiple reps within a trial
            imed = plot.get_imed(Ts) # picking out median rep in trial
            for i,ys in enumerate(yss):
                if i == imed:
                    vals = ys
                    if len(labels)>0:
                        label = labels[l] # label is for the current parameter value, as opposed to param_label which is the name of the parameter
                    else:
                        label = ''
                    if '=' not in label: label = label.title()
                    if boxplot or scatterplot:
                        if '=' in label and '(' not in label:
                            param,v = labels[l].split('=') # i.e. pick out specific value v to use as ticklabel
                            v = v.strip()
                        else:
                            v = label # e.g. labels are network kinds, so use specific network kind as the ticklabel
                        if scatterplot:
                            pos = float(v)
                            wid = 0.01
                            col = (0,0,0)
                            valss+=list(vals)
                            posx = [pos]*len(vals)
                            poss+=posx
                        else:
                            k+=1
                            pos = k
                            wid = 0.8
                            col = colours[k]
                        pdict = dict(color=col)
                        if boxplot:
                            violin = ax.violinplot(ys,positions=[pos],showextrema=False,widths=[wid])
                            for part in violin['bodies']: # can't set colour directly when making violin
                                part.set_facecolor(col)
                                part.set_edgecolor(col)
                            ax.boxplot(vals,positions=[pos],labels=[v],widths=[wid],capprops=pdict,flierprops=dict(marker='x',mec=col,mfc=col),boxprops=pdict,whiskerprops=pdict,medianprops={**pdict,'linestyle':'--'})
                        if scatterplot:
                            ax.plot(posx,vals,'k.') # overlay on top of boxplot
                    else:
                        spl=letters[l]
                        colour = colours[l]
                        if '=' not in xlabel: xlabel = xlabel.title()
                        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
                        counts,edges = np.histogram(vals,bins)
                        pcs = 100*counts/sum(counts)
                        ax.stairs(pcs, edges, fill=True,color=colour,label=label) # plot without using matplotlib hist so can control bar heights to be percentages
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel('Frequency (%)')
                        ax.set_xlim(*xlim)
                        ax.set_ylim(0,100)
                        ax.annotate('Mean: %.2f'%np.mean(vals),xy=(0.88,0.935),xycoords='figure fraction',va='bottom',ha='right',color=colour)
                        ax.annotate('Median: %.2f'%np.median(vals),xy=(0.88,0.9),xycoords='figure fraction',va='bottom',ha='right',color=colour)
                        ax.legend()
                        name = plot.make_name(labels[l]+'_'+xlabel)
                        file = 'hist_'+name
                        plot.add_spl(ax,spl)
                        if folder=='':
                            plt.show()
                        else:
                            plot.savefig(os.path.join(folder,file))
                            plt.close()
        ax.grid(which='major',axis='both')
        if log:
            ax = plot.set_log(ax,vals)
        if boxplot or scatterplot:
            ax.set_xlabel(param_label)
            ax.set_ylabel(xlabel)
            if scatterplot:
                ax = plot.add_trendline(ax,xs=poss,ys=valss,deg=1)
            ax = plot.try_rotation(ax)
            plot.add_spl(ax,spl)
            name = plot.make_name(param_label+'_'+xlabel)
            file = 'dist_'+name
            plot.savefig(os.path.join(folder,file))
            plt.close()        
        return
    def plot_scatter(ip,param_combos,Ts,labels,sens_param_labels,Tname,polydeg=None,boxplot=False,log=True,spl='',folder=''):
        def are_text(lis):
            return [isinstance(l,str) for l in lis]
        '''
        scatter plot of dependence of response variable Ts against sensitivity parameters in param_combos, with trendline fitted as polynomial of suitable degree
        if sens_param is nominal, plots box plot instead (since there is no suggestion of continuous x-direction behaviour)
        i is parameter index of given sensitivity parameter in param_combos
        param_combos is list of tuples [(sensitvity param values) for each trial]
        Ts is list of lists [[trial0 reps],[trial1 reps]], where each sublist has a single value for each rep
        sens_param_names is the name of each sensitivity parameter (only need ith one)
        '''
        colours = plot.get_base_colours()
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        xlabel = sens_param_labels[ip]
        xss = []; yss = []
        for i,(combo,Ts_) in enumerate(zip(param_combos,Ts)):
            # combo contains parameter value for ith sensitivity case
            x = combo[ip] # parameter value of parameter being plotted
            if isinstance(x,list):
                val = x[0]
            elif isinstance(x,str):
                if '=' in x:
                    _,val = x.split('=') # trying to pick out numerical value
                    val = val.strip()
            try:
                x = float(val) # if val can't be made float, x will stay as a string
            except:
                pass
            # Ts_ contains the take-off times (or any other metric with a single value per rep) for ith sensitivity case
            ys = Ts_
            if boxplot or isinstance(x,str) or isinstance(x,bool) or isinstance(x,list):
                boxplot = True
                fullwid = 0.8
                gap = 1-fullwid
                col = colours[i]
                pdict_ = dict(color=col)
                lab = labels[i].title() if '$' not in labels[i] else labels[i]
                if len(np.shape(ys))>1:
                    ncollect = np.shape(ys)[0] # i.e. number of collections of y to plot as adjacent boxplots
                    yss = ys
                else:
                    ncollect = 1
                    yss = [ys]
                wid = fullwid/ncollect
                shifts = np.linspace(0,fullwid-wid,ncollect)
                shifts-=gap/ncollect
                for j,ys in enumerate(yss):
                    pos = i+shifts[j]
                    labh = lab if j==0 else ''
                    pdict = {**pdict_,'linewidth':2} if ncollect==3 and j==1 else pdict_
                    violin = ax.violinplot(ys,positions=[pos],showextrema=False,widths=[wid])
                    for part in violin['bodies']: # can't set colour directly when making violin
                        part.set_facecolor(col)
                        part.set_edgecolor(col)
                    ax.boxplot(ys,positions=[pos],labels=[labh],widths=[wid],capprops=pdict,flierprops=dict(marker='x',mec=col,mfc=col),boxprops=pdict,whiskerprops=pdict,medianprops={**pdict,'linestyle':'--'})
            else:
                boxplot = False
                xs = [x for _ in range(len(ys))] # repeating the parameter value to match the number of dependent values
                ax.plot(xs,ys,'k.')
                xss+=xs
                yss+=ys
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Tname)
        if boxplot:
            labelsh = [x.split('=')[-1].strip() if len(x.split('='))>1 else x for x in labels]
            for i in range(len(labelsh)):
                val = labelsh[i]
                try:
                    val = int(val)
                    integer = True
                except ValueError:
                    integer = False
                if not integer:
                    try:
                        val = float(val)
                        real = True
                    except ValueError:
                        real = False
                if not integer and not real:
                    if '$' not in val:
                        val = val.replace(' ','\n')
                labelsh[i] = val
            ax.set_xticks(fullwid/2-gap+np.arange(len(labels)), labelsh)
            if all(are_text(labelsh)):
                if any([len(tl)>6 and len(labelsh)>3 for tl in labelsh]) and not any(['$' in tl for tl in labelsh]): # $ indicates shorter label
                    plt.xticks(rotation=40,ha='right')
        else:
            if not any(are_text(xss)): # i.e. have numerical parameter so can plot trendline
                ax = plot.add_trendline(ax,xs=xss,ys=yss,deg=polydeg)
        ax.grid(which='major',axis='both')
        if log:
            ylim = ax.get_ylim()
            ax = plot.set_log(ax,ylim)
        plot.add_spl(ax,spl)
        name = plot.make_name(Tname+'_'+xlabel)
        pref = 'boxplot_' if boxplot else 'scatter_'
        plot.savefig(os.path.join(folder,pref+name))
        plt.close()
        return
    def plot_corr(ziss,zss,Ts_,cats,cols,spl='',folder=''): # zss,ziss,cats=labels,cols=['Social Diffusion (%)','Information Flow (%)',param_label]
        def make_frame(lls,cats,cols): # take list-of-lists lls of single depth (e.g. zs for median r for multiple cases), where each sublist corresponds to a category in cats, and make a dataframe with column headers cols
            df = pd.DataFrame()
            vals = [] # values to correlate
            cvals = [] # category to distinguish vals by
            for ll,Ts,cat in zip(lls,Ts_,cats):
               imed = plot.get_imed(Ts) 
               vals += list(ll[imed]) # lls is (cases,reps,times), so ll is (reps,times)
               cat_ = cat.split('=')[1].strip() if '=' in cat else cat
               cvals += [cat_]*len(ll[imed])
            df[cols[0]] = vals # note that cols here doesn't quite match cols in the calling function (it's a subset)
            df[cols[1]] = cvals
            return df
        dfzi = make_frame(100*np.array(ziss),cats,[cols[0],cols[2]])
        dfz = make_frame(100*np.array(zss),cats,[cols[1],cols[2]])
        df = dfz.join(dfzi,rsuffix='_r').drop(columns=cols[2]+'_r')
        colours = plot.get_base_colours()
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        ax = sns.scatterplot(df,x=cols[0],y=cols[1],hue=cols[2],palette=colours,ax=ax)
        ax = plot.add_trendline(ax,xs=df[cols[0]].to_list(),ys=df[cols[1]].to_list(),deg=1)
        ax.set_xlim((0,100))
        ax.set_ylim((0,100))
        plot.add_spl(ax,spl)
        ax = plot.try_rotation(ax)
        name = plot.make_name('corr_'+'_diff_v_flow')
        plot.savefig(os.path.join(folder,name))
        plt.close()
        return
    def plot_bars(ecss,ind,Ts_,param_label,labels,ylabel='Connection Count',frac=False,log=True,spl='',folder=''): # ecss is specifically either adjacency or trust (i.e. not both, even though the original list contains both)
        def make_frame(ll,case='',case_label='',ylabel='',hue='',cats=[r'Co$\rightarrow$Ex',r'Co$\rightarrow$La',r'Ex$\rightarrow$Ex',r'Ex$\rightarrow$La',r'La$\rightarrow$Ex',r'La$\rightarrow$La']): # take flat list of values for [comexp,comlag,expexp,explag,lagexp,laglag] for median rep of case, and put into dataframe
            df = pd.DataFrame()
            df[ylabel] = ll
            df[hue] = cats
            df[case_label] = case.split('=')[1].strip() if '=' in case else case # single value the same for all cats
            return df
        dfs = []
        hue = 'Connection Type'
        for ecs,Ts,lab in zip(ecss,Ts_,labels): # ecss is (cases,reps,2 for adj and trust,6 connection types)
            imed = plot.get_imed(Ts)
            df = make_frame(ecs[:][imed][ind],lab,param_label,ylabel,hue)
            if frac: df[ylabel]/=df[ylabel].sum()
            dfs.append(df)
        df = pd.concat(dfs,ignore_index=True)
        colours = ['black','grey','red','tomato','royalblue','cornflowerblue']
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        ax = sns.barplot(df,x=param_label,y=ylabel,hue=hue,palette=colours,ax=ax)
        if not frac:
            ax = plot.set_log(ax,df[ylabel])
        ax = plot.make_legend(ax)
        ax = plot.try_rotation(ax)
        plot.add_spl(ax,spl)
        name = plot.make_name('bars_'+ylabel)
        plot.savefig(os.path.join(folder,name))
        plt.close()
        return
    def plot_network(A, props=[], title = '', description='Follower Count', suffix='', layout='', cmap='viridis', nticks=5, nxplot=True, cb=True, vmin=None, vmax=None, spl='', with_labels=False, folder=''): # pyvis allows more options and interactivity, but exports only to html
        # if layout='', pyvis randomly locates nodes using spring physics, else if 'circ', impose regular circular layout without physics, else any other string, impose random layout without physics
        def get_colours(values,cmap):
            # convert values to colours, see https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
            vals = values[~np.isnan(values)]
            norm = mpc.Normalize(min(vals), max(vals), clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            colours = [None if np.isnan(v) else mapper.to_rgba(v) for v in values]
            return mapper,colours
        def make_colourbar(values,description='',cmap='viridis',nticks=5,dp=2,filename=''):
            # create standalone colorbar and save to file
            fig = plt.figure()
            a_0,ma = [min(values),max(values)]
            left, bottom, width, height = [0., 0., 1., 0.1] # dimensions for axes object
            ax = fig.add_axes([left, bottom, width, height])
            mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap)
            ticks = list(np.linspace(a_0,ma,nticks))
            dummy_ticks = list(np.linspace(left,left+width,nticks)) # dummy_ticks are the tick positions in the figure coordinates
            fmt=fmt='%.'+str(int(dp))+'f'
            ax.set_xticks(dummy_ticks, [fmt%x for x in ticks])
            ax.set_title(description)
            plot.savefig(filename, dpi='figure') # note that the folder is included in the filename
            plt.close()
            return
        n = len(A)
        # define properties from adjacency matrix if none given
        if len(props)==0: props = np.sum(A,axis=1) # sum of sent values from each node (i.e. 'follower count' of each node)
        # determine whether props are discrete / define number of decimal places to show node values to
        dp = 0 if all([int(prop)==prop for prop in props]) else 2 # 0 is effectively a flag for discrete int values, otherwise dp is number of decimal places to show node values to
        # use networkx to create network with node colour defined by props (node sizes and edge widths all constant)
        gx = network.get_graph(A)
        if layout=='circ':
            pos = nx.circular_layout(gx,scale=n**2) # force a regular circular layout (have to do this with networkx and then pass it to pyvis)
        else:
            pos = None
        node_values = np.array([props[int(k)] for k in gx.nodes]) # because of how nodes are defined from edges, their sequencing varies, so need to be careful to match node values with node ids
        if nxplot: # plot using networkx (based on matplotlib); see https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html
            fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
            if vmin==None: vmin = min(node_values)
            if vmax==None: vmax = max(node_values)
            if vmin==vmax:
                vmin = 0.9*vmin; vmax = 1.1*vmax # this is how matplotlib will rescale the colour bar
            node_size=300/n if n>20 else 300
            nx.draw_networkx(gx,pos,with_labels=with_labels,node_color=node_values,node_size=node_size,cmap=cmap,vmin=vmin,vmax=vmax,ax=ax)
            ax.set_axis_off()
            ax.set_title(title)
            ax.axis('equal')
            if cb: # see https://stackoverflow.com/questions/49572089/how-to-get-a-colorbar-in-networkx-draw-networkx
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array(props)
                cbar = plt.colorbar(sm,shrink=0.7,ax=ax)
                cbar.ax.set_ylabel(description)
            plot.add_spl(ax,spl)
            if folder=='':
                plt.show()
            else:
                plot.savefig(os.path.join(folder,'network_'+title+suffix))
                plt.close()
            return
        # otherwise, plot using pyvis
        mapper,colours = get_colours(node_values,cmap) # colours only used in pyvis plot
        gp = Network(notebook=True,directed=True)
        gp.from_nx(gx)
        tempname='temp.html'
        for node in gp.get_nodes():
            k = int(gp.get_node(node)['id'])
            if not np.isnan(node_values[k]):
                # commented out to keep node size constant: node['size'] = node_values[k] # i.e. the sum of sent values from node k gives the node size
                gp.get_node(node)['color'] = mpc.to_hex(colours[k]) # i.e. " " colour
                #gp.get_node(node)['label'] = ''
                gp.get_node(node)['shape'] = 'circle' # circle allows node labels inside, see https://pyvis.readthedocs.io/en/latest/_modules/pyvis/network.html#Network.add_node
                if layout!='': # see https://stackoverflow.com/questions/74108243/pyvis-is-there-a-way-to-disable-physics-without-losing-graphs-layout
                    gp.get_node(node)['physics']=False
                    if layout=='circ':
                        gp.get_node(node)['x'] = pos[node][0]
                        gp.get_node(node)['y'] = -pos[node][1]
            else:
                del node
        for edge in gp.edges:
            i = int(edge['from'])
            j = int(edge['to'])
            # commented out to keep edge width constant: edge['width'] = A[i,j] # i.e. the value sent from i to j gives the arrow width
        if layout!='': gp.toggle_physics(False)
        gp.show(tempname)
        # make standalone colourbar as image file
        cfilename = os.path.join(folder, 'network_'+description.replace(' ','_')+suffix+'_colourbar.png')
        title = description
        make_colourbar(node_values,description=title,cmap=cmap,nticks=nticks,dp=dp,filename=cfilename)
        # add colourbar to html file
        filename = cfilename.replace('_colourbar.png','.html')
        with open(tempname,'r') as read_f, open(filename,'w') as write_f:
            write_f.write('<img src="'+os.path.basename(cfilename)+'">'+'\n')
            for line in read_f:
                write_f.write(line)
        os.remove(tempname)
        return
    def try_polys(xs,ys,mind=1,maxd=1):
        best_r2 = 0; best_deg = 0; best_z = 0 # by definition, these will be improved upon at least once in the loop
        for deg in range(mind,maxd+1): # +1 due to 0-indexing quirk on upper limits
            # fit polynomial to given deg
            z = np.polyfit(xs,ys,deg) # z contains the polynomial coefficients for the fitted curve
            p = np.poly1d(z)
            pxs = p(xs) if deg>1 else xs # i.e. if linear model, don't compare linear fit to itself
            reg = scipy.stats.linregress(pxs,ys) # linear regression of fitted curve to data
            r2 = reg.rvalue**2 # this will be the linear model fit if deg = 1, else the goodness of fit for higher deg polynomial
            if r2>best_r2:
                best_r2 = r2
                best_z = z
                best_deg = deg
        p = np.poly1d(best_z) # p is the fitted polynomial function obtained from best_z    
        return p,best_r2,best_deg
    def add_trendline(ax,xs,ys,deg=None,col=None): # use trendline with given deg of polynomial; if deg=None, best fitting deg will be found
        if len(xs)==0 or len(ys)==0: # nothing to regress
            return ax
        # fit polynomial (search through deg fits to get best one)
        if deg==None:
            mind = 1
            maxd = len(list(set(xs)))-1 # impose well-defined polynomial
        else:
            mind = deg
            maxd = deg
        if deg==None:
            poly,r2_value,deg = plot.try_polys(xs,ys,mind=mind,maxd=maxd)
        if deg>1: # plot polynomial curve (don't bother if linear, as will show this below)
            # plot fitted curve
            npoints = 100 if deg>1 else 2 # number of x values to use to plot polynomial curve
            xvals = np.linspace(min(xs),max(xs),npoints) # x values used to print polynomial curve
            if col==None:
                colour = 'b'
            else:
                colour = col
            ax.plot(xvals,poly(xvals),ls='-',color=colour)
            # display goodness of fit for fitted curve, i.e. compare p(xs) with ys
            pref = '$P_{'+str(deg)+'}$ ' if deg>1 else '' # only want to show order of polynomial if greater than 1
            ax.annotate(pref+'$R^2$ = %.2f'%r2_value,xy=(0.8,0.8),xycoords='figure fraction',ha='right',color=colour)
        # regardless of deg, display linear trend as r value for xs and ys
        lr = scipy.stats.linregress(xs,ys)
        if col==None:
            colour = 'r'
            ls = '-'
        else:
            colour = col
            ls = '--'
        ax.plot([min(xs),max(xs)],[lr.intercept+lr.slope*min(xs),lr.intercept+lr.slope*max(xs)],color=colour,ls=':')
        ax.annotate('$R$ = %.2f'%lr.rvalue,xy=(0.74,0.74),xycoords='figure fraction',ha='right',color=colour)
        return ax

if __name__=='__main__':
    plotG = False # test network plotting
    plotCD = False # test connectivity distribution curve
    plotBR = False # test best-response function
    plotHM = False # test plotting multiple lines as heatmap
    getSP = False # test shortest path
    getAS = True # test generating network and obtaining clustering metric
    
    if plotG:
        n = 10
        folder = file.make_folder('results','networks')
        g = 2 # power law distribution shape
        a_0 = 3/n # corresponding minimum activity value
        degss = []
        labels = ['complete', 'regular', 'small world', 'scale free', 'directed', 'homophilous']
        for ki,label in enumerate(labels):
            props = []
            hs = []
            network_params = {}
            if label=='complete':
                pass
            elif label=='regular':
                network_params = dict(d=0.3)
            elif label=='small world':
                network_params = dict(k=0.3,p=0.1) # k nearest neighbours
            elif label=='scale free':
                network_params = dict(m=0.3)
            elif label=='directed':
                network_params = dict(g=g,a_0=a_0)
            elif label=='homophilous':
                hs = np.random.rand(n)
                props = hs.copy() # hs is used in get_adjacencies, props is used in plot_network
                network_params = dict(g=g,a_0=a_0)
            print(label)
            A = network.get_adjacencies(n,kind=label,network_params=network_params,hs=hs,seed=np.random.randint(10000))
            if len(hs)==0:
                description = 'Follower Count'
            else:
                description = 'Agent Property'
            title = label.title()
            for key,val in network_params.items():
                title += ', $'+str(key)+'$ = '+str(val)
            spl = letters[ki]
            plot.plot_network(A,props=props,title=title,description=description,layout='circ',nxplot=True,spl=spl,folder=folder)
            degs = analyse.get_degrees(A)
            degss.append([degs]) # putting in format expected by plot_dist
        Ts = [[0] for _ in degss] # again, putting into format expected by plot_dist
        #plot.plot_dist(degss,Ts,['Topology'],labels,xlabel='Node degree',folder=folder)
        # last A obtained in loop is for homophilous network, so analyse it further
        H = [] # this will be list of abs difs in props for i and j
        C = [] # corresponding list for number of connections for i and j
        for i in range(n):
            for j in range(n):
                if j<i:
                    dif = abs(props[i]-props[j])
                    H.append(dif)
                    C.append(A[i,j]+A[j,i]) # sum both possible connections for i and j
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        ax.plot(H,C,'k.')
        ax.set_xlabel('Magnitude of Property Difference')
        ax.set_ylabel('Number of Connections')
        ax = plot.add_trendline(ax,xs=H,ys=C,deg=1)
        plot.add_spl(ax,'b')
        plot.savefig(os.path.join(folder,'correlation'))
        plt.close()

    if plotCD:
        folder = file.make_folder('results','networks')
        colours = plot.get_base_colours()
        n = 10
        a_0=3/n  # minimum connectivity of any agent  
        a_0_ = a_0/2
        gs = [2,1.5] # gamma parameter values (can't have gamma==1)
        xs = np.linspace(a_0,1,100) # connectivities to plot scalefree_pdf for
        xs_ = np.linspace(a_0_,1,100) # connectivities to plot scalefree_pdf for
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        for i,g in enumerate(gs):
            plt.plot(xs,[network.scalefree_pdf(x,g=g,a_0=a_0) for x in xs],color=colours[i],label='$g$=%.1f'%g+', $a_0$=%.2f'%a_0)
            plt.plot(xs_,[network.scalefree_pdf(x,g=g,a_0=a_0_) for x in xs_],color=colours[i],ls=':',label='$g$=%.1f'%g+', $a_0$=%.2f'%a_0_)
        plt.legend()
        plt.xlabel('Follower Count as Fraction of Agents')
        plt.ylabel('Probability Density')
        plt.xlim([0,1])
        plot.add_spl(ax,'a')
        plot.savefig(os.path.join(folder,'prob_dist'))
        plt.close()

    if plotBR:
        E0s = np.linspace(0,1,3)
        betas = np.linspace(0,1,4)
        E1 = np.linspace(0,1,100)
        for E0 in E0s:
            fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
            for beta in betas:
                plt.plot(E1,np.exp(beta*E1)/(np.exp(beta*E1)+np.exp(beta*E0)),label='$\\beta$ = %.1f'%beta)
            plt.xlabel('$E(x=1)$')
            plt.ylabel('$P(x=1)$')
            plt.ylim([0,1])
            plt.title('$E(x=0)$ = %.1f'%E0)
            plt.legend()
    
    if getSP:
        #A = network.get_adjacencies(10,'directed')
        A = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,1],[1,0,1,1]])
        plot.plot_network(A,layout='circ',with_labels=True,show=True)
        sps = analyse.get_sps(A,dev=True)
        lsps = list(sps)
        #print(lsps)
        lengths = [] # note the directions are accounted for, i.e. it is sent length
        for node in lsps:
            ls = list(node[1].values()) # node[0] is node the id, node[1] is the dictionary of path lengths
            ls = ls[1:] if len(ls)!=1 else [0] # ignore 1st entry which is distance to self (0 by definition)
            lengths+=ls # lengths is a flat list of distances between all nodes
            #print(node[0],ls)
        print(lengths)
        
    if plotHM:
        zss_ = np.array([[[.1,.2,.3,.4,.2],[.1,.3,.4,.2,.3],[.1,.5,.4,.6,.7]],[[.2,.4,.6,.6,.4],[.2,.3,.6,.3,.7],[.3,.3,.4,.3,.5]]])
        Ts_ = np.array([[1,2,5],[45,22,34]])
        ts_pt = 5
        labels = ['Case 1','Case 2']
        arr = plot.plot_timeseries(zss_,Ts_,labels=labels,ts_pt=ts_pt,case_label='Case',heat=True,declutter=False,spl='',folder='')
        
    if getAS:
        # n = 5
        # label = 'small world'
        # network_params = dict(k=0.8,p=0.1)
        # A = network.get_adjacencies(n,network_params=network_params,kind=label,seed=np.random.randint(10000))
        A = np.array([[[0],[1],[1],[0]],[[0],[0],[1],[0]],[[0],[1],[0],[1]],[[1],[0],[0],[0]]])
        plot.plot_network(A,layout='circ',nxplot=True,cmap='spring',with_labels=True)
        clusters = analyse.get_clusters(A)
        print(clusters)