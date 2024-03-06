# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:48:43 2023

@author: mark.pogson@liverpool.ac.uk

this script is used to define variables for a specific trial and then run it using social_diffusion.py

"""

import social_diffusion
import itertools

# define variables ------------------------------------------------------------
# give descriptive name for results subfolder (end with _ as date and time will be appended)
trial_description = '3_status_'

# make definitions for plotting
plot_timeseries = True # fraction of active agents with strategy 1 over time
plot_network = False # with large n (e.g. above a few dozen), this may be of limited use
plot_hist = True # histogram of switching rates for agent in median rep for trial
plot_scatter = True # n.b. there must be more than one sensitivity case for at least one parameter for any scatter plots to be made
plot_bars = True # plot grouped bars for inter-group edge counts/trust sums
plot_corr = True # plot correlation scatter for social and information diffusion

# define parameter names (order must be consistent with other lists below)
Tname = 'Takeoff Time' # used to identify time metric to use in plotting

# define agent numbers and parameter values (note that agent parameter b will be calculated from k and r)
nrep = 25 # number of reptitions for each sensitivity case
n = 200 # number of agents

# define the network (need to come back to this for sensitivity analysis as their parameters are specific to the network topology)
network_kinds = ['scale free'] # although this is a list of sensitivity cases, it's treated differently to the other sensitivity parameters; options: 'complete', 'directed', 'homophilous', 'small world', 'regular', 'scale free' (see functions.network.get_adjacencies)   
nwpks = [] # network parameter keys (must be empty if more than one network kinds, or must match with network kind; use empty list for no network parameters)

# define model/analysis parameters not to include in the sensitivity analysis
tmax=1000 # number of time steps to iterate over
frac_takeoff=0.2 # fraction of active agents changed from starting strategy required for take-off
frac_diffusion=0.9 # " " full diffusion
fixed_network = False # if False, a new network is generated each time with different random effects
period = 'pre' # 'full' to obtain switching rate across full time period; 'pre' for up to diffusion time; 'post' for after diffusion time; 'inter' for between takeoff and diffusion time
    
# define parameter values for sensitivity analysis (all combinations of these will be used - i.e. not using latin hypercube sampling on continuous ranges)
consolidateds = [False] # True to apply group_tfs to explorers and followers (in that order), else use aft to spread tfs evenly across all active agents
stratifieds = [False,True] # True to impose weighted influence based on follower count
nwp0s = [None] # vals to correspond with 1st key in nwpks, e.g. if nwpks=['k','p'], you'd list the sensitivity cases of p here
nwp1s = [None] # " " 2nd " "
nwps = [[nwp0,nwp1] for nwp0,nwp1 in zip(nwp0s,nwp1s)]
frac_c0s = [0.]  # fraction of agents committed to strategy 0 relative to frac_c1, i.e. frac_c0 doesn't make sense on its own
frac_c1s = [0.25] # " " 1
frac_es = [0.2] # fraction of explorers among non-committed agents
beta_es = [7.8] # rationality for explorers
beta_fs = [7.8] # " " followers
k_es = [0.10] # inertia for explorers
k_fs = [0.42] # " " followers
r_es = [0.42] # trend-seeking for explorers
r_fs = [0.16] # " " followers

# define relative group sizes as ratios
af0s = [1] # if one_way=False, af0 gives the reciprocal of the fraction of active agents with initial strategy 0, e.g. af0=3 means 1 in 3 agents will have initial strategy 0
afts = [1] # like af0, but for group_tfs, so always needs defining (aft=1 means only the 1st group td value will be used)
group_tf0s = [1] # time difference parameter to use for trend-seeking in 1st agent group
group_tf1s = [1] # " " 2nd " " (set as group_tf0s to make uniform)
group_tfs = [[group_tf0,group_tf1] for group_tf0,group_tf1 in zip(group_tf0s,group_tf1s)]

# define timeseries and histogram plot labels based on sensitivity parameters, i.e. those with more than 1 value listed - labels need to reflect combinations of parameter values if more than one is varied
merges = [True] # True to make committed agent properties match active agents for homophily
shuffles = [True,False] # True to make all agents randomly sequenced to avoid preferential attachment/small world issues
stras = ['Stratified' if s else 'Unstratified' for s in stratifieds]
pris = [' Unprioritised' if sh else ' Prioritised' for sh in shuffles]
combos = list(itertools.product(stras,pris))
labels = [s + p for s,p in combos]

# run simulation --------------------------------------------------------------
social_diffusion.run(trial_description,
        plot_timeseries,plot_network,plot_hist,plot_scatter,plot_bars,plot_corr,
        Tname,
        nrep,n,
        nwpks,
        tmax,frac_takeoff,frac_diffusion,fixed_network,period,
        network_kinds,consolidateds,stratifieds,nwps,frac_c0s,frac_c1s,frac_es,beta_es,beta_fs,k_es,k_fs,r_es,r_fs,
        af0s,group_tfs,afts,labels,merges=merges,shuffles=shuffles)