# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:30:52 2023

@author: mark.pogson@liverpool.ac.uk

replot results from social_diffusion model outputs and recalculate metrics

"""

import os
import glob
import pickle
import functions
import itertools
import numpy as np
import string

letters = string.ascii_lowercase

plot_timeseries = True # fraction of active agents with strategy 1 over time
plot_network = False # with large n (e.g. above a few dozen), this may be of limited use
plot_hist = True # histogram of switching rates for agent in median rep for trial
plot_scatter = True # n.b. there must be more than one sensitivity case for at least one parameter for any scatter plots to be made
plot_bars = True # plot grouped bars for inter-group edge counts/trust sums
plot_corr = True # plot correlation scatter for social and information diffusion
read_metrics = True # True to read in, False to recalculate from raw model outputs

def load(fold,file):
    with open(os.path.join(fold,file,file+'.pkl'),'rb') as f:
        data = pickle.load(f)
    return data

do = ['status']
frac_takeoff=0.2 # fraction of active agents changed from starting strategy required for takeoff
frac_diffusion=0.9 # " " full diffusion
fixed_network = False # if False, a new network is generated each time with different random effects
period = 'pre' # 'full' to obtain switching rate across full time period; 'pre' for up to diffusion time; 'post' for after diffusion time; 'inter' for between takeoff and diffusion time

for trial in glob.glob(os.path.join('results','*')):
    if 'store' not in trial and 'replotted' not in trial and (len(do)==0 or any([x in trial for x in do])): # i.e. trial is listed for replotting
        print(trial)
        fold = os.path.join(trial,'data')
        try:
            xall = load(fold,'xall')
            css = load(fold,'css')
            ess = load(fold,'ess')
            As = load(fold,'As')
        except:
            print('data missing for '+trial)
            continue
        if read_metrics:
            try:
                print('...reading metrics')
                Tks = load(fold,'Tks') # these can be reobtained based on xall and css (and ess to get all,explorer,laggard separately)
                Tds = load(fold,'Tds')
                TkIs = load(fold,'TkIs')
                TdIs = load(fold,'TdIs')
                ecss = load(fold,'ecss')
                zss = load(fold,'zss')
                ziss = load(fold,'ziss')
                spss = load(fold,'spss')
                try:
                    clustss = load(fold,'clustss')
                except:
                    clustss = None
            except:
                print('data missing for '+trial)
                continue
        else:
            # get metrics (allows for changes/updates in the calculations)
            TkIs=[];TdIs=[];Tks=[];Tds=[];yss=[];zss=[];ziss=[];spss=[];ecss=[];clustss=[]
            print('...calculating metrics from raw model outputs...')
            for case,(xall__,cs_,es_,As_) in enumerate(zip(xall,css,ess,As)): # each case
                print('...case',case,'of',len(xall))
                TkIs.append([]);TdIs.append([]);Tks.append([]);Tds.append([]);yss.append([]);zss.append([]);ziss.append([]);spss.append([]);ecss.append([]);clustss.append([])
                for r,(xall_,cs,es,A) in enumerate(zip(xall__,cs_,es_,As_)): # each rep
                    Tk,Td,ys,zs,TkI,TdI,zis = functions.analyse.get_metrics(xall_,A,cs,es=es,period=period,frac_takeoff=frac_takeoff,frac_diffusion=frac_diffusion)
                    TkIs[-1].append(TkI)
                    TdIs[-1].append(TdI)
                    Tks[-1].append(Tk)
                    Tds[-1].append(Td)
                    yss[-1].append(ys)
                    zss[-1].append(zs)
                    ziss[-1].append(zis)
                    sps = functions.analyse.get_sps(A)
                    spss[-1].append(sps)
                    ecs = functions.analyse.get_edges(A,cs,es)
                    ecss[-1].append(ecs)
                    clusts = functions.analyse.get_clusters(A)
                    clustss[-1].append(clusts)
            # (over)write to original folder so can be replotted again by reading in, which is quicker
            foldh = os.path.split(fold)[0] # removing data subfolder from fold as writing files automatically adds this, and want to overwrite in same place
            functions.file.pickle_data(yss,'yss',folder=foldh)
            functions.file.pickle_data(zss,'zss',folder=foldh)
            functions.file.pickle_data(ziss,'ziss',folder=foldh)
            functions.file.pickle_data(Tks,'Tks',folder=foldh)
            functions.file.pickle_data(Tds,'Tds',folder=foldh)
            functions.file.pickle_data(TkIs,'TkIs',folder=foldh)
            functions.file.pickle_data(TdIs,'TdIs',folder=foldh)
            functions.file.pickle_data(spss,'spss',folder=foldh)
            functions.file.pickle_data(ecss,'ecss',folder=foldh)
            functions.file.pickle_data(clustss,'clustss',folder=foldh)
            functions.file.to_csv(zss,'zss',folder=foldh)
            functions.file.to_csv(ziss,'ziss',folder=foldh)
            functions.file.to_csv(Tks,'Tks',folder=foldh)
            functions.file.to_csv(Tds,'Tds',folder=foldh)
            functions.file.to_csv(TkIs,'TkIs',folder=foldh)
            functions.file.to_csv(TdIs,'TdIs',folder=foldh)
        
        if clustss==None:
            print('...calculating clustering coefficients from raw model outputs...')
            clustss=[]
            for As_ in As:
                clustss.append([])
                for A in As_:
                    clusts = functions.analyse.get_clusters(A)
                    clustss[-1].append(clusts)
            foldh = os.path.split(fold)[0]
            functions.file.pickle_data(clustss,'clustss',folder=foldh)
            
        # make folder for outputs and define parameter values
        folder = functions.file.make_folder(trial+'_replotted')         
        if 'topology' in trial:
            network_kinds = ['complete', 'regular', 'small world', 'scale free', 'directed', 'homophilous'] # although this is a list of sensitivity cases, it's treated differently to the other sensitivity parameters; options: 'complete', 'directed', 'homophilous', 'small world', 'regular', 'scale free' (see functions.network.get_adjacencies)   
            labels = [k.title() for k in network_kinds]
            param_combos = network_kinds
            sens_param_labels = ['']
        elif 'valency' in trial:
            ms = [0.005,0.01,0.05,0.1,0.2]
            labels = ['$m$ = '+str(m) for m in ms]
            param_combos = [[m] for m in ms]
            sens_param_labels = ['$m$']
        elif 'timescale' in trial:
            ts = [1,10,50,100]
            labels = ['$t_{\\delta}$ = %d'%t for t in ts]
            param_combos = [[t] for t in ts]
            sens_param_labels = ['$t_{\\delta}$']
        elif 'consolidated' in trial or 'homophily' in trial:
            gts = [[1,1],[1,100],[100,1]]
            labels = ['$t_{\\delta}$=('+str(t[0])+','+str(t[1])+')' for t in gts]
            param_combos = [[gt] for gt in gts]
            sens_param_labels = ['$t_{\\delta}$']
        elif 'consolidation' in trial or 'heterogeneity' in trial:
            afts = [1,2,3,5]
            labels = ['$\\tau$ = %d'%tg for tg in afts]
            param_combos = [[aft] for aft in afts]
            sens_param_labels = ['$\\tau$']
        elif 'status' in trial:
            stratifieds = [False,True]
            shuffles = [True,False] # True to make all agents randomly sequenced to avoid preferential attachment/small world issues
            stras = ['Stratified' if s else 'Unstratified' for s in stratifieds]
            pris = ['\nUnprioritised' if sh else '\nPrioritised' for sh in shuffles]
            combos = list(itertools.product(stras,pris))
            labels = [s + p for s,p in combos]
            param_combos = [[combo] for combo in combos]
            sens_param_labels = ['']
                
        if plot_timeseries:
            print('...plotting timeseries')
            Tdsh = functions.analyse.get_flat(Tds,0) # extract takeoff time for all active agents from list of [[active,explorer,laggard],...]
            #spl = '' if 'topology' in trial else 'a'
            functions.plot.plot_timeseries(zss,Tdsh,labels=labels,spl='a',folder=folder)
            #functions.plot.plot_timeseries(zss,Tdsh,labels=labels,spl='b',heat=True,folder=folder)
            functions.plot.plot_timeseries(zss,Tdsh,labels=labels,heat=True,spl='b',folder=folder)
            functions.plot.plot_timeseries(ziss,Tdsh,labels=labels,ylabel='Information Diffusion (%)',spl='c',folder=folder)
            functions.plot.plot_timeseries(ziss,Tdsh,labels=labels,ylabel='Information Diffusion (%)',spl='d',heat=True,folder=folder)
        
        if plot_hist or plot_scatter or plot_corr:
            Tksh = functions.analyse.get_flat(Tks,[1,0,2]) # reorder takeoff times so explorers, combined, laggards
            Tdsh = functions.analyse.get_flat(Tds,[1,0,2]) # diffusion " "
            Tsh = functions.analyse.get_flat(Tds,0) # combined diffusion times only
            TkIsh = functions.analyse.get_flat(TkIs,[1,0,2]) # reorder takeoff times so explorers, combined, laggards
            TdIsh = functions.analyse.get_flat(TdIs,[1,0,2]) # diffusion " "
            param_label = sens_param_labels[0] if 'topology' not in trial and 'status' not in trial else ''
            k = 0

            try:
                degss = load(fold,'degss')
                #spss = load(fold,'spss') # calculated this with get metrics loop, so can use updated function
            except:
                print('data missing for '+trial)
                continue

            if plot_hist: # distribution of agent values within median rep
                print('...plotting distributions')
                if 'valency' in trial:
                    boxplot = False
                    scatterplot = True
                else:
                    boxplot = True
                    scatterplot = False # don't add trendline if x-axis not used numerically
                    if 'topology' in trial:
                        k = 3
                functions.plot.plot_dist(degss,Tsh,param_label=param_label,labels=labels,xlabel='Node Valency',log=True,boxplot=boxplot,scatterplot=scatterplot,spl=letters[k],folder=folder)
                k+=1
                functions.plot.plot_dist(spss,Tsh,param_label=param_label,labels=labels,xlabel='Shortest Path Length',log=False,boxplot=boxplot,scatterplot=scatterplot,spl=letters[k],folder=folder)
                if 'status' in trial:
                    k = 1
                else:
                    k+=1
                functions.plot.plot_dist(clustss,Tsh,param_label=param_label,labels=labels,xlabel='Clustering Coefficient',log=False,boxplot=boxplot,scatterplot=scatterplot,spl=letters[k],folder=folder)

            if plot_scatter: # plot of single pre-level metric (e.g. takeoff time) from each rep (c.f. plot_hist which is distribution of agent values (e.g. node valency) within median rep)        
                print('...plotting scatters')
                ip = 0
                k = 0
                functions.plot.plot_scatter(ip,param_combos,Tksh,labels,sens_param_labels,'Takeoff Time',polydeg=1,boxplot=True,spl=letters[k],folder=folder)
                if 'status' not in trial:
                    k+=1
                functions.plot.plot_scatter(ip,param_combos,Tdsh,labels,sens_param_labels,'Diffusion Time',polydeg=1,boxplot=True,spl=letters[k],folder=folder)
                if 'valency' in trial:
                    k+=1
                    functions.plot.plot_scatter(ip,param_combos,TkIsh,labels,sens_param_labels,'Information Takeoff Time',polydeg=1,boxplot=True,spl=letters[k],folder=folder)
                k+=1
                functions.plot.plot_scatter(ip,param_combos,TdIsh,labels,sens_param_labels,'Information Diffusion Time',polydeg=1,boxplot=True,spl=letters[k],folder=folder)
                # k+=1
                # degrs = [[np.mean(degs_) for degs_ in degs] for degs in degss]
                # functions.plot.plot_scatter(ip,param_combos,degrs,labels,sens_param_labels,'Mean Node Degree',polydeg=1,spl=letters[k],folder=folder)
                # k+=1
                # spprs = [[np.mean(sps_) for sps_ in sps] for sps in spss]
                # functions.plot.plot_scatter(ip,param_combos,spprs,labels,sens_param_labels,'Mean Shortest Path Length',polydeg=1,spl=letters[k],folder=folder)
                
            if plot_corr:
                print('...plotting correlations')
                k = 3
                functions.plot.plot_corr(ziss,zss,Tsh,cats=labels,cols=['Information Flow (%)','Social Diffusion (%)',param_label],spl=letters[k],folder=folder)
        
        if plot_bars:
            print('...plotting bars')
            k = 0
            if 'status' in trial:
                k = 2
            functions.plot.plot_bars(ecss,0,Tsh,param_label,labels,ylabel='Connection Count',spl=letters[k],folder=folder) # ecss is: [[comexp,comlag,expexp,explag,lagexp,laglag],[same again but using trust not adjacency]]
            k+=1
            functions.plot.plot_bars(ecss,1,Tsh,param_label,labels,ylabel='Trust Sum',spl=letters[k],folder=folder)
            functions.plot.plot_bars(ecss,0,Tsh,param_label,labels,ylabel='Connection Fraction',frac=True,folder=folder) # ecss is: [[comexp,comlag,expexp,explag,lagexp,laglag],[same again but using trust not adjacency]]
            functions.plot.plot_bars(ecss,1,Tsh,param_label,labels,ylabel='Trust Fraction',frac=True,folder=folder)
