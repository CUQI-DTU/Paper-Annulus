###################################################
# This code was used to produce the results in 
# paper-draft: "Bayesian reconstruction of geometric 
# parameters that characterize subsea pipes using CT 
# measurements" that was included in the PhD thesis
# of Silja L. Christensen
# May 2024
###################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import dill
import cuqi

#%%=======================================================================
# Choises
#=========================================================================

no_layers = 1

resultpath = '../../../../../../work3/swech/results/'
resultname = 'OneAnnulus_CC_std01_Mixed'

datapath = './synthdata/'
dataname = 'OneAnnulus_CC_std01'

plotpath = './plots/'
os.makedirs(plotpath, exist_ok=True)

# Sample analysis
Nb = 4000
Nt = 1
included_chain_nos = np.array([0,1,2,3,4,5,6,7,8,9])

# plot specifications
cmin = -0.2
cmax = 1
cmap_image = "gray"
cmap_lines = "tab10"
if no_layers == 1:
    paramnames = ["Center x", "Center y", "Inner radius", "Width", "Attenuation"]
    paramshortnames = ["x", "y", "r", "w", "phi"]
elif no_layers == 4:
    paramnames = ["Center x", "Center y", "Inner radius", "Width 1", "Attenuation 1", "Width 2", "Attenuation 2", "Width 3", "Attenuation 3", "Width 4", "Attenuation 4"]
    paramnames2 = ["Center x", "Center y", "Inner radius", "Width", "Attenuation"]
    paramshortnames = ["x", "y", "r", "w1", "phi1", "w2", "phi2", "w3", "phi3", "w4", "phi4"]

#%%=======================================================================
# load data
#=========================================================================
with open('{}{}.pkl'.format(resultpath, resultname), 'rb') as f:  # Python 3: open(..., 'rb')
    samples, Org, A = dill.load(f)

with open('{}{}.pkl'.format(datapath, dataname), 'rb') as f:  # Python 3: open(..., 'rb')
    theta_true, d_obs, noise_std = dill.load(f)

domain = 4
no_starts = len(samples)
Ns = samples["start{0}".format(0)].Ns
pipe_geom = A.domain_geometry

print(Org.paramlist)

# %%=======================================================================
# process and plot samples
# =========================================================================

#%% Truth and data
# sinogram
fig, ax = plt.subplots(1,1, figsize=(5,3))
cs = d_obs.plot(aspect = 1/2, extent = [0, 300, 360, 0], interpolation = "none", cmap = cmap_image)
cs[0].axes.set_xticks(np.linspace(0,300, 5, endpoint = True))
cs[0].axes.set_yticks(np.linspace(0,360, 7, endpoint = True))
cs[0].axes.set_xlabel('Detector')
cs[0].axes.set_ylabel('View angle [degree]')
fig.subplots_adjust(right=0.85, bottom=0.15)
cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
cbar = plt.colorbar(cs[0], cax=cax)
plt.savefig(plotpath + resultname + '_sinogram.png')

# ground_truth
fig, ax = plt.subplots(1,1, figsize=(4,3))
cs = theta_true.plot(origin = "lower", interpolation = "none", extent = [-domain/2, domain/2, -domain/2, domain/2], vmin = cmin, vmax = cmax, cmap = cmap_image)
cs.axes.set_xticks(np.linspace(-domain/2,domain/2, 5, endpoint = True))
cs.axes.set_yticks(np.linspace(-domain/2,domain/2, 5, endpoint = True))
fig.subplots_adjust(right=0.95)
cax = fig.add_axes([cs.axes.get_position().x1+0.01,cs.axes.get_position().y0,0.03,cs.axes.get_position().height])
cbar = plt.colorbar(cs, cax=cax)
plt.savefig(plotpath + resultname + '_ground_truth_image.png')

fig, ax = plt.subplots(1,1, figsize=(4,2))
theta_true.plot(plot_par = True)
plt.savefig(plotpath + resultname + '_ground_truth_params.png')

#%% Multichains
colmap_chains = plt.get_cmap(cmap_lines)
for k in range(A.domain_dim):
    fig, ax = plt.subplots(1,1, figsize=(4,2))
    for i in range(no_starts):
        samples["start{0}".format(i)].plot_chain(k, lw = 0.8, ls = '-', color = colmap_chains(i))
    ax.plot(np.array([0,Ns]), np.array([theta_true[k], theta_true[k]]), lw = 1.2, ls ='--', color = 'k')
    ax.set_ylabel(paramnames[k], fontsize = 10)
    ax.set_xlabel("Realization no.", fontsize = 10)
    ax.get_legend().remove()
    fig.subplots_adjust(left = 0.18, bottom = 0.22)
    plt.savefig(plotpath + resultname + '_chains_' + paramshortnames[k]+'.png')

colmap_chains = plt.get_cmap(cmap_lines)
for k in range(A.domain_dim):
    fig, ax = plt.subplots(1,1, figsize=(4,2))
    for i in range(no_starts):
        samples["start{0}".format(i)].burnthin(Nb).plot_chain(k, lw = 0.8, ls = '-', color = colmap_chains(i))
    ax.plot(np.array([0,Ns-Nb]), np.array([theta_true[k], theta_true[k]]), lw = 1.2, ls ='--', color = 'k')
    ax.set_ylabel(paramnames[k], fontsize = 10)
    ax.set_xlabel("Realization no.", fontsize = 10)
    ax.set_xticks(np.arange(0, Ns-Nb+1, 1000))
    ax.set_xticklabels(np.arange(Nb, Ns+1, 1000))
    ax.get_legend().remove()
    fig.subplots_adjust(left = 0.18, bottom = 0.22)
    plt.savefig(plotpath + resultname + '_chainsburnin_' + paramshortnames[k]+'.png')

# Burnthin and merge chains 
samples_merged =  cuqi.samples.Samples(samples["start{0}".format(0)].samples, geometry=pipe_geom, is_par=True).burnthin(Nb, Nt = Nt).samples
for i in included_chain_nos[1:]:
    tmp = cuqi.samples.Samples(samples["start{0}".format(i)].samples, geometry=pipe_geom, is_par=True).burnthin(Nb, Nt = Nt)
    samples_merged = np.hstack([samples_merged, tmp.samples])

samples_merged = cuqi.samples.Samples(samples_merged, geometry=pipe_geom, is_par=True)

#%% chain correlation
fig, ax = plt.subplots(1,1, figsize=(4,2))
samples_merged.plot_autocorrelation(variable_indices=range(pipe_geom.par_shape[0]), max_lag=None, combined=False)
plt.savefig(plotpath + resultname + '_acf.png')

ess = samples_merged.compute_ess()
print(ess)

#%% pair plot
plt.figure()
cs = samples_merged.plot_pair(marginals = False, textsize = 30)
plt.savefig(plotpath + resultname + '_marginals.png')

#%% plot stats
lo_conf, up_conf = samples_merged.compute_ci(95)
mean = samples_merged.mean()
std = np.sqrt(samples_merged.variance())
print(mean)
print(std)

# mean and ci of parameters
if no_layers > 1:
    colors = plt.get_cmap(cmap_lines)
    fig, axs = plt.subplots(1, 1, figsize=(8,3))
    x = np.array([0, 1.5, 3, 4.5, 6])
    xx = np.linspace(-no_layers/2*0.18,no_layers/2*0.18,no_layers, endpoint=True)
    axs.plot(np.array([x[0]-0.7, x[-1]+0.7]), np.zeros(2), lw = 1.2, ls ='--', color = 'k')
    axs.errorbar(x[0:3], mean[0:3]-theta_true[0:3], yerr=np.vstack((mean[0:3]-lo_conf[0:3], up_conf[0:3]-mean[0:3])), fmt ='o', capsize =3, ms = 3, color = colors(0))
    for i in range(no_layers):
        axs.errorbar(x[3:]+xx[i], mean[i+3::no_layers]-theta_true[i+3::no_layers], yerr=np.vstack((mean[i+3::no_layers]-lo_conf[i+3::no_layers], up_conf[i+3::no_layers]-mean[i+3::no_layers])), fmt ='o', capsize =3, ms = 3, color = colors(i), label = "Layer {}".format(i+1))
    axs.set_ylabel("Posterior mean - truth")
    axs.set_xticks(x)
    axs.set_xticklabels(paramnames2)
    plt.legend()
    plt.savefig(plotpath + resultname + '_ci.png')

else:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,3))
    ax.axhline(0, lw = 1.2, ls ='--', color = 'k')
    ax.errorbar(paramnames, lo_conf-theta_true, yerr=np.vstack((np.zeros(len(lo_conf)), up_conf-lo_conf)),
                                color = 'dodgerblue', fmt = 'none' , capsize = 3, capthick = 1, label = "95% CI")
    ax.plot(paramnames, mean-theta_true, 's', color = 'tab:blue', label = "Mean - truth")
    plt.legend()
    plt.savefig(plotpath + resultname + '_ci.png')

