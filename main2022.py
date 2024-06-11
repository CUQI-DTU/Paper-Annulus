#%%
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
import os

import annulus_geometry2022 as annulus_geometry

# GitLab CUQI
sys.path.append('../cuqipy/')
import cuqi
from cuqi.astra.model import FanBeam2DModel

# %load_ext autoreload
# %autoreload 2

#%%=======================================================================
# Choises
#=========================================================================

resultpath = './output/test/'
os.makedirs(resultpath, exist_ok=True)


#%%=======================================================================
# Define annulus parameters (ground_truth can be defined now or later)
#=========================================================================
norings = 2
annulus_geom_type = "ConcentricConnected_annuli"

param0 = annulus_geometry.Annulus_Param("center_x", 0, value = 0.1, prior=cuqi.distribution.Uniform(low = -1, high = 1))
param1 = annulus_geometry.Annulus_Param("center_y", 0, value = 0.2, prior=cuqi.distribution.Uniform(low = -1, high = 1))
param2 = annulus_geometry.Annulus_Param("inner_r", 0, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.6))
param3 = annulus_geometry.Annulus_Param("width", 0, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.7))
param4 = annulus_geometry.Annulus_Param("abscoeff", 0, value = 0.7, prior=cuqi.distribution.Uniform(low = 0.5, high = 0.9))
param5 = annulus_geometry.Annulus_Param("width", 1, value = 0.3, prior=cuqi.distribution.Uniform(low = 0.1, high = 0.5))
param6 = annulus_geometry.Annulus_Param("abscoeff", 1, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.6))
# param7 = annulus_geometry.Annulus_Param("width", 2, prior=cuqi.distribution.Uniform(low = 0.1, high = 0.5))
# param8 = annulus_geometry.Annulus_Param("abscoeff", 2, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.6))
# param9 = annulus_geometry.Annulus_Param("width", 3, prior=cuqi.distribution.Uniform(low = 0, high = 0.4))
# param10 = annulus_geometry.Annulus_Param("abscoeff", 3, prior=cuqi.distribution.Uniform(low = 0.6, high = 1))

# colorbar lims when plotting
cmin = -0.2
cmax = 1

annulus_params_list = [param0, param1, param2, param3, param4, param5, param6]

#%%=======================================================================
# Define acqusition geometry
#=========================================================================

# Scan Geometry parameters
DetectorCount = 300 # no of detectors
domain = 4
AngleCount = 100 # no of view angles
maxAngle = 2*np.pi
theta = np.linspace(0,maxAngle,AngleCount, endpoint=True)
source_object_dist = 7
object_detector_dist = 5
det_spacing = 10/DetectorCount
m = DetectorCount * AngleCount

#%%=======================================================================
# Create synthetic data
#=========================================================================
# Reconstruction grid
Nphantom = 1024

# Model
model_phantom = FanBeam2DModel(im_size = (Nphantom,Nphantom),
                                det_count = DetectorCount,
                                angles = theta,
                                source_object_dist = source_object_dist,
                                object_detector_dist = object_detector_dist,
                                det_spacing = det_spacing,
                                domain = (domain,domain),
                                proj_type = "cuda")

# Geometry
phantom_geometry = annulus_geometry.ConcentricConnected_annuli(norings=norings, imagesize=domain, pixeldim = Nphantom, c_coords='cartesian')
model_phantom.domain_geometry = phantom_geometry

# Setup phantom from annulus params
phantom = phantom_geometry.annulusparams2paramvec(annulus_params_list)
cuqi_phantom = cuqi.samples.CUQIarray(phantom, is_par=True, geometry = phantom_geometry)

# Evaluate the exact data
b_true = model_phantom(cuqi_phantom)

# Add noise
rnl = 0.10
e0 = np.random.normal(0, 1, np.shape(b_true))
noise_std = rnl*np.linalg.norm(b_true)/np.linalg.norm(e0)
data_dist_phantom  = cuqi.distribution.GaussianSqrtPrec(mean = model_phantom, sqrtprec = sp.sparse.diags(1/noise_std*np.ones(m), format = 'csc'), geometry=model_phantom.range_geometry)
np.random.seed(10)
b_data=data_dist_phantom(x=cuqi_phantom).sample()

#%%=======================================================================
# Setup reconstruction problem
#=========================================================================

# Reconstruction grid
N = 500

# Model
model_recon = FanBeam2DModel(im_size = (N,N),
                                det_count = DetectorCount,
                                angles = theta,
                                source_object_dist = source_object_dist,
                                object_detector_dist = object_detector_dist,
                                det_spacing = det_spacing,
                                domain = (domain,domain),
                                proj_type = "cuda")

# Geometry
recon_geometry = annulus_geometry.ConcentricConnected_annuli(norings=norings, imagesize=domain, pixeldim = N, c_coords='cartesian')
model_recon.domain_geometry = recon_geometry

# Define ground truth with recon geometry
ground_truth = cuqi.samples.CUQIarray(phantom, is_par=True, geometry = recon_geometry)

# Likelihood
data_dist = cuqi.distribution.GaussianCov(mean = model_recon, cov=noise_std**2, geometry=model_recon.range_geometry)
likelihood = data_dist.to_likelihood(b_data)

# Prior
prior = recon_geometry.setup_prior(annulus_params_list = annulus_params_list, geometry=recon_geometry)

# Posterior
posterior = cuqi.distribution.Posterior(prior = prior, likelihood = likelihood)

#%%=======================================================================
# Setup sampler
#=========================================================================

Ns = 10500      # no of samples in each chain
Nb = 1000       # Burnin
Nt = 50         # Thinning
sample_scale = 1e-3 # Initial sample scale

# setup initial guess and transform to CUQIarray
x0 = prior.sample(1)

# setup CUQI sampler
sampler = cuqi.sampler.CWMH(posterior, scale = sample_scale, x0 = x0)

#%%=======================================================================
# Sample Posterior
#=========================================================================

samples = sampler.sample_adapt(N = Ns, Nb = 0)

#%%=======================================================================
# save data
#=========================================================================

# with open('{}samples.pkl'.format(resultpath), 'wb') as f:  # Python 3: open(..., 'wb')
#     dill.dump([samples, Ns, Nb, Nt, sample_scale, annulus_params_list, ground_truth, annuli_geometry, b_data, data_geometry, noise_std, rnl, proj_geom], f)

#%%=======================================================================
# Test of load saved data
#=========================================================================

# with open('{}samples.pkl'.format(resultpath), 'rb') as f:  # Python 3: open(..., 'rb')
#     samples1, Ns1, Nb1, Nt1, sample_scale1, paramnames1, paramfilenames1, annulus_params_list1, ground_truth1, annuli_geometry1, b_data1, data_geometry1, noise_std1, rnl1, proj_geom1 = dill.load(f)

# b_data1.geometry = data_geometry1
# b_data1.is_par = True
# ground_truth1.geometry = annuli_geometry1
# ground_truth1.is_par = True

#%%=======================================================================
# process and plot samples
#=========================================================================

# sinogram
plt.figure()
b_data.plot(aspect = "auto", interpolation = "none")
plt.colorbar()
plt.savefig(resultpath + 'sinogram.png')

# ground_truth
plt.figure()
ground_truth.plot(origin = "lower", interpolation = "none", vmin = cmin, vmax = cmax)
plt.colorbar()
plt.savefig(resultpath + 'ground_truth.png')

# plot chains
plt.figure()
samples.plot_chain(variable_indices = range(3+norings*2))
plt.savefig(resultpath + 'chains.png')

# burnin
Nb = 5500
samples_burnin= samples.burnthin(Nb, Nt = 1)

# plot autocorrelations
plt.figure()
samples_burnin.plot_autocorrelation(variable_indices=range(3+norings*2), max_lag=None, combined=False)
plt.savefig(resultpath + 'acf.png')

# burnthin
samples_burnthin = samples.burnthin(Nb, Nt = Nt)

# plot chains
plt.figure()
samples_burnthin.plot_chain(variable_indices = range(3+norings*2))
plt.savefig(resultpath + 'chains_burnthin.png')

# chains/hist
plt.figure()
samples_burnthin.plot_trace(variable_indices=range(3+norings*2)) 
plt.savefig(resultpath + '1Dmarginals.png')

# stats
plt.figure()
samples_burnthin.plot_mean(origin = "lower", interpolation = "none", vmin = cmin, vmax = cmax) 
plt.colorbar()
plt.savefig(resultpath + 'mean.png')

plt.figure()
samples_burnthin.plot_ci(plot_par = True, exact = ground_truth) # crashes without plot_par = True
plt.savefig(resultpath + 'ci.png')

# matrix plot
plt.figure()
samples_burnthin.plot_pair(variable_indices=range(3+norings*2)) 
plt.savefig(resultpath + '2Dmarginals.png')



# %%
