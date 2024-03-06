# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:02:49 2023

@author: mark.pogson@liverpool.ac.uk

this script calculates social diffusion based on the model of Ye et al. (2021)
it implements this on specified network topologies
it is called from a control script specific to a particular model trial
it uses functions.py for calculations, plotting and data export

"""
import functions
import numpy as np
from datetime import datetime
import itertools
import string

def run(trial_description,
        plot_timeseries,plot_network,plot_dist,plot_scatter,plot_bars,plot_corr,
        Tname,
        nrep,n,
        nwpks,
        tmax,frac_takeoff,frac_diffusion,fixed_network,period,
        network_kinds,consolidateds,stratifieds,nwps,frac_c0s,frac_c1s,frac_es,beta_es,beta_fs,k_es,k_fs,r_es,r_fs,
        af0s,group_tfs,afts,labels,merges=[True],shuffles=[True],ts_spl='',ts_pt=200,declutter=False,write_data=True):  

    # set up variables --------------------------------------------------------
    # print start time
    t0 = datetime.now()
    print(trial_description+' simulations started at', t0)
      
    fixed_param_names = ['nrep','n','network_kind','nwpks','tmax','frac_takeoff','frac_diffusion','fixed_network','consolidated','stratified','period']
    sens_param_names = ['network_kinds','consolidateds','stratifieds','nwps','frac_c0s','frac_c1s','frac_es','beta_es','beta_fs','k_es','k_fs','r_es','r_fs','af0s','group_tfs','afts','shuffles','merges']
    sens_param_labels = ['Topology','Consolidated','Social Status','$m$','$r_{\\rho0}$','$\\rho_{c1}$','$\\rho_e$','$\\beta_e$','$\\beta_f$','$k_e$','$k_f$','$r_e$','$r_f$','$\\sigma$','$t_{\\delta}$','$\\tau$','Prioritised','Merged']
    param_names = fixed_param_names + sens_param_names

    letters = string.ascii_lowercase # used for subplot labels
        
    # obtain all possible combinations of defined parameter values
    sens_params = [network_kinds,consolidateds, stratifieds, nwps, frac_c0s, frac_c1s, frac_es, beta_es, beta_fs, k_es, k_fs, r_es, r_fs, af0s, group_tfs, afts, merges, shuffles] # this ordering needs to be maintained (haven't used dictionary for simpicity)
    param_combos = list(itertools.product(*sens_params))
    
    # obtain list of sens params which are actually varied
    sens_param_labelsh = [l for s,l in zip(sens_params,sens_param_labels) if len(s)>1]
    param_label = ', '.join(sens_param_labelsh) # name of sensitivity case parameter(s) in combo, to be used as axis label in dist plot

    # put fixed parameters in list to save
    fixed_params = [nrep,n,nwpks,tmax,frac_takeoff,frac_diffusion,fixed_network,period]

    # write parameter values to file
    folder = functions.file.make_folder('results',trial_description+t0.strftime('%Y_%m_%d_%Hh%Mm%S'))
    functions.file.write_params(fixed_params+sens_params,param_names,folder)
    
    # run simulation ----------------------------------------------------------
    # initialise lists for results in the form [[trial0 reps],[trial1 reps],...]    
    TkIs=[];TdIs=[];Tks=[];Tds=[];yss=[];zss=[];ziss=[];xalls=[];As=[];degss=[];spss=[];css=[];ess=[];ecss=[] # take-off times, diffusion times, switching counts, fraction of agents with strategy 1, all agent strategies at all time points, adjacency matrix, node degrees, shortest path lengths between all pairs of nodes, explorer agents out of active agents
    # run simulations for the trials and reps
    for i,param_combo in enumerate(param_combos):
        print('Sensitivity case ',i+1,'/',len(param_combos))
        TkIs.append([]);TdIs.append([]);Tks.append([]);Tds.append([]);yss.append([]);zss.append([]);ziss.append([]);xalls.append([]);As.append([]);degss.append([]);spss.append([]);css.append([]);ess.append([]);ecss.append([])
        # obtain diffusion time, transition time, switching rates for each agent and fraction of agents with strategy 1 at each time step
        for rep in range(nrep):
            # create lists of parameter values for network, as well as each agent and parameter
            # use n and cs to create adjacency matrix
            # obtain parameters at each rep (as may be shuffled - get_paramss will deal with this)
            stratified,network_kind,network_params,betas,ks,rs,bs,ss,cs,es,tfs = functions.initialise.get_params(n,nwpks,*param_combo,case=i,rep=rep,folder=folder)
            css[-1].append(cs)
            ess[-1].append(es)
            if i==0 or not fixed_network: # only need to create network for first trial; subsequent trials will use same network
                if len(network_kinds)>1: # i.e. sensitivity just for network topology
                    hs = bs # i.e. use social coordination parameter to provide homophily for homophilous network
                else: # i.e. only looking at a single network topology, so use timescale parameter for homophily (the timescale parameter would need already to be set to be different for different groups)
                    hs = tfs
                seed = rep + i*nrep
                A = functions.network.get_adjacencies(n,kind=network_kind,network_params=network_params,hs=hs,seed=seed) # n.b. network parameters are distinct from agent parameters
                A[:,cs] = 0 # making committed agents unable to receive
                A = functions.network.get_trusts(A,stratified) # A is now normalised trust matrix obtained from adjacency matrix if stratified=True, else it's just the adjacency matrix renamed for consistency
                As[-1].append(A)
                degs = functions.analyse.get_degrees(A)
                degss[-1].append(degs)
                sps = functions.analyse.get_sps(A)
                spss[-1].append(sps)
                ecs = functions.analyse.get_edges(A,cs,es)
                ecss[-1].append(ecs)
            xall = functions.simulate.diffuse(A,betas,ks,rs,bs,ss,cs,tfs,tmax=tmax)
            xalls[-1].append(xall)
            frac_c0 = param_combo[0]
            frac_c1 = param_combo[1]
            Tk,Td,ys,zs,TkI,TdI,zis = functions.analyse.get_metrics(xall,A,cs,es=es,period=period,frac_takeoff=frac_takeoff,frac_diffusion=frac_diffusion)
            TkIs[-1].append(TkI)
            TdIs[-1].append(TdI)
            Tks[-1].append(Tk)
            Tds[-1].append(Td)
            yss[-1].append(ys)
            zss[-1].append(zs)
            ziss[-1].append(zis)
    if write_data:
        functions.file.pickle_data(xalls,'xall',folder=folder)
        functions.file.pickle_data(yss,'yss',folder=folder)
        functions.file.pickle_data(zss,'zss',folder=folder)
        functions.file.pickle_data(ziss,'ziss',folder=folder)
        functions.file.pickle_data(css,'css',folder=folder)
        functions.file.pickle_data(ess,'ess',folder=folder)
        functions.file.pickle_data(As,'As',folder=folder)
        functions.file.pickle_data(Tks,'Tks',folder=folder)
        functions.file.pickle_data(Tds,'Tds',folder=folder)
        functions.file.pickle_data(Tks,'TkIs',folder=folder)
        functions.file.pickle_data(Tds,'TdIs',folder=folder)
        functions.file.pickle_data(degss,'degss',folder=folder)
        functions.file.pickle_data(spss,'spss',folder=folder)
        functions.file.pickle_data(ecss,'ecss',folder=folder)
        functions.file.to_csv(xalls,'xall',folder=folder)
        functions.file.to_csv(zss,'zss',folder=folder)
        functions.file.to_csv(ziss,'ziss',folder=folder)
        functions.file.to_csv(css,'css',folder=folder)
        functions.file.to_csv(ess,'ess',folder=folder)
        functions.file.to_csv(As,'As',folder=folder)
        functions.file.to_csv(Tks,'Tks',folder=folder)
        functions.file.to_csv(Tds,'Tds',folder=folder)
        functions.file.to_csv(TkIs,'TkIs',folder=folder)
        functions.file.to_csv(TdIs,'TdIs',folder=folder)    
    # plot time series of percentage of agents adopting strategy 1
    if Tname == 'Diffusion Time':
        Ts = Tds # time metric to use to find medians for plotting
    elif Tname == 'Takeoff Time':
        Ts = Tks
    else:
        print('Error: time metric not recognised for plotting')
    Tsh = functions.analyse.get_flat(Ts,0) if len(np.shape(Ts))>2 else Ts # i.e. dealing with potentially separate values for all agents, explorers and laggards - see get_metrics()
    Tksh = functions.analyse.get_flat(Tks,0) if len(np.shape(Tks))>2 else Tks # i.e. dealing with potentially separate values for all agents, explorers and laggards - see get_metrics()
    Tdsh = functions.analyse.get_flat(Tds,0) if len(np.shape(Tds))>2 else Tds # i.e. dealing with potentially separate values for all agents, explorers and laggards - see get_metrics()
    TkIsh = functions.analyse.get_flat(TkIs,0) if len(np.shape(TkIs))>2 else TkIs # i.e. dealing with potentially separate values for all agents, explorers and laggards - see get_metrics()
    TdIsh = functions.analyse.get_flat(TdIs,0) if len(np.shape(TdIs))>2 else TdIs # i.e. dealing with potentially separate values for all agents, explorers and laggards - see get_metrics()
    if plot_timeseries: # rep-level curves for rep-level temporal metric, e.g. fraction of active agents with strategy 1
        # plot timeseries of each rep of each sensitivity case, with median of each sensitivity case highlighted
        spl_ts = ts_spl if 'timescale' not in trial_description else 'a'
        spl_hm = ts_spl if 'timescale' not in trial_description else 'b'
        ts_pth = ts_pt if ts_pt<=len(zss[0][0]) else len(zss[0][0]) # time limit for plot
        # limited time
        functions.plot.plot_timeseries(zss,Tsh,labels=labels,ts_pt=ts_pth,declutter=declutter,spl=spl_ts,folder=folder) # plot zs for each rep, where rep is highlighted according to median value in Tsh
        functions.plot.plot_timeseries(zss,Tsh,labels=labels,ts_pt=ts_pth,heat=True,case_label=sens_param_labelsh[0],declutter=declutter,spl=spl_hm,folder=folder) # plot zs for each rep, where rep is highlighted according to median value in Tsh
        # full time
        functions.plot.plot_timeseries(zss,Tsh,labels=labels,ts_pt=None,declutter=declutter,spl=spl_ts,folder=folder) # plot zs for each rep, where rep is highlighted according to median value in Tsh
        functions.plot.plot_timeseries(zss,Tsh,labels=labels,ts_pt=None,heat=True,case_label=sens_param_labelsh[0],declutter=declutter,spl=spl_hm,folder=folder) # plot zs for each rep, where rep is highlighted according to median value in Tsh
        # limited time - information diffusion
        functions.plot.plot_timeseries(ziss,Tsh,labels=labels,ts_pt=ts_pth,ylabel='Information Flow (%)',declutter=declutter,spl=spl_ts,folder=folder) # plot zis for each rep, where rep is highlighted according to median value in Tsh
        functions.plot.plot_timeseries(ziss,Tsh,labels=labels,ts_pt=ts_pth,ylabel='Information Flow (%)',heat=True,case_label=sens_param_labelsh[0],declutter=declutter,spl=spl_hm,folder=folder) # plot zis for each rep, where rep is highlighted according to median value in Tsh
        # full time - information diffusion
        functions.plot.plot_timeseries(ziss,Tsh,labels=labels,ts_pt=None,ylabel='Information Flow (%)',declutter=declutter,spl=spl_ts,folder=folder) # plot zis for each rep, where rep is highlighted according to median value in Tsh
        functions.plot.plot_timeseries(ziss,Tsh,labels=labels,ts_pt=None,ylabel='Information Flow (%)',heat=True,case_label=sens_param_labelsh[0],declutter=declutter,spl=spl_hm,folder=folder) # plot zis for each rep, where rep is highlighted according to median value in Tsh
    if plot_dist: # case-level distribution for agent-level metric of median rep, e.g. switching rate of agents
        if 'valency' in trial_description:
            boxplot = False
            scatterplot = True
        else:
            boxplot = True
            scatterplot = False # don't add trendline if x-axis not used numerically
        # plot distribution of agent switching rates for median of each sensitivity case
        #functions.plot.plot_dist(yss,Tsh,param_label=param_label,labels=labels,xlabel='Switching Rate',boxplot=boxplot,scatterplot=scatterplot,spl='c',folder=folder) # plot ys for rep which has median value in Tsh
        # " " node degrees " "
        functions.plot.plot_dist(degss,Tsh,param_label=param_label,labels=labels,xlabel='Node Valency',boxplot=boxplot,scatterplot=scatterplot,spl='a',folder=folder)
        # " " shortest path lengths between all pairs of nodes " "
        functions.plot.plot_dist(spss,Tsh,param_label=param_label,labels=labels,xlabel='Shortest Path Length',boxplot=boxplot,scatterplot=scatterplot,spl='b',folder=folder)
    if plot_network: # plot median network for each rep
        # " " network " "
        for i,Ts_ in enumerate(Tsh): # n.b. Ts_ is the rep values in Ts (reversed notation from functions.plot.plot_dist)
            if i==0 or not fixed_network:
                imed = functions.plot.get_imed(Ts_) # picking out median rep in trial
                if fixed_network: imed = 0 # this occurs if i==0 and fixed_network==True
                functions.plot.plot_network(As[i][imed],title=labels[i],layout='circ',nxplot=True,spl=letters[i],folder=folder)
    if plot_scatter: # rep-level plot of single metric        
        # plot scatter of rep level diffusion times and mean shortest path lengths for each parameter in sensitivity analysis
        k = -1
        for ip,param in enumerate(sens_params):
            if len(param)>1: # i.e. more than one sensitivity value for parameter
                # median node degree per rep vs param sensivity
                #k+=1
                #degrs = [[np.mean(degs_) for degs_ in degs] for degs in degss]
                #functions.plot.plot_scatter(ip,param_combos,degrs,labels,sens_param_labels,'Mean Node Degree',polydeg=1,spl=letters[k],folder=folder)
                # median shortest path length per rep vs param sensivity
                #k+=1
                #spprs = [[np.mean(sps_) for sps_ in sps] for sps in spss]
                #functions.plot.plot_scatter(ip,param_combos,spprs,labels,sens_param_labels,'Mean Shortest Path Length',polydeg=1,spl=letters[k],folder=folder)
                # take-off time per rep vs param sensivity
                k+=1
                functions.plot.plot_scatter(ip,param_combos,Tksh,labels,sens_param_labels,'Takeoff Time',polydeg=1,spl=letters[k],folder=folder)
                # diffusion time per rep vs param sensivity
                k+=1
                functions.plot.plot_scatter(ip,param_combos,Tdsh,labels,sens_param_labels,'Diffusion Time',polydeg=1,spl=letters[k],folder=folder)
                # information flow take-off time per rep vs param sensivity
                k+=1
                functions.plot.plot_scatter(ip,param_combos,TkIsh,labels,sens_param_labels,'Information Takeoff Time',polydeg=1,spl=letters[k],folder=folder)
                # information flow diffusion time per rep vs param sensivity
                k+=1
                functions.plot.plot_scatter(ip,param_combos,TdIsh,labels,sens_param_labels,'Information Diffusion Time',polydeg=1,spl=letters[k],folder=folder)
    if plot_corr:
        functions.plot.plot_corr(ziss,zss,Tsh,cats=labels,cols=['Information Flow (%)','Social Diffusion (%)',param_label],folder=folder)
    if plot_bars:
        functions.plot.plot_bars(ecss,0,Tsh,param_label,labels,ylabel='Connection Count',folder=folder) # ecss is: [[comexp,comlag,expexp,explag,lagexp,laglag],[same again but using trust not adjacency]]
        functions.plot.plot_bars(ecss,1,Tsh,param_label,labels,ylabel='Trust Sum',folder=folder)
    # print runtime
    t1 = datetime.now()
    print('simulations finished at', t1)
    print('total runtime = ', t1-t0)