###################################################
# Main script for running experiments with geometric parameterization of pipes
# By Silja L. Christensen
# June 2024
###################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import dill

from AnnulusGeometry2024 import PipeParam, PipeParamsCollection, DiskFree, DiskConcentric, AnnulusFree, AnnulusConcentricConnected
# cuqipy version 1.0.0
from cuqi.distribution import Gaussian, Gamma, Uniform, JointDistribution
from cuqi.sampler import CWMH
from cuqi.experimental.mcmc import CWMHNew, HybridGibbsNew, MHNew
from cuqi.likelihood import Likelihood
from cuqi.array import CUQIarray
# cuqipy-cil version 0.6.0
from cuqipy_cil.model import FanBeam2DModel, ShiftedFanBeam2DModel
# CIL version 22.1
from cil.utilities.display import show_geometry

import subprocess
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

#%%=======================================================================
# Paths
#=========================================================================

# path for saving results
resultpath = '../../../../../../work3/swech/results/'
resultname = 'GibbsTest'
os.makedirs(resultpath, exist_ok=True)

#%%=======================================================================
# Discretization
#=========================================================================
N = 500
N_phantom = 1024
imagesize = 4

#%%=======================================================================
# Parameter lib
#=========================================================================

# DiskFree
DF1 = PipeParam(paramtype = "center_x", 
                layerno = 0,
                truevalue = -0.1,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF2 = PipeParam(paramtype = "center_y", 
                layerno = 0,
                truevalue = 0.2,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF3 = PipeParam(paramtype = "radius", 
                layerno = 0, 
                truevalue = 0.4,
                prior=Uniform(low = 0.3, high = 0.5))
DF4 = PipeParam(paramtype = "center_x", 
                layerno = 1,
                truevalue = -0.1,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF5 = PipeParam(paramtype = "center_y", 
                layerno = 1,
                truevalue = 0.2,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF6 = PipeParam(paramtype = "radius", 
                layerno = 1, 
                truevalue = 0.9,
                prior=Uniform(low = 0.7, high = 1.1))
DF7 = PipeParam(paramtype = "abscoeff", 
                layerno = 1,
                truevalue = 0.7,
                prior=Gamma(shape = 2, rate = 2))
DF8 = PipeParam(paramtype = "center_x", 
                layerno = 2,
                truevalue = 0,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF9 = PipeParam(paramtype = "center_y", 
                layerno = 2,
                truevalue = 0.1,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
DF10 = PipeParam(paramtype = "radius", 
                layerno = 2, 
                truevalue = 1.1,
                prior=Uniform(low = 0.9, high = 1.3))
DF11 = PipeParam(paramtype = "abscoeff", 
                layerno = 2,
                truevalue = 0.3,
                prior=Gamma(shape = 2, rate = 2))


#%%=======================================================================
# Parameter lib
#=========================================================================

nolayers = 2

pipeparams_list = [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10, DF11]

pipe_geometry = DiskFree(nolayers, imagesize, N)

# Collect the info above in one object
PPCollection = PipeParamsCollection(pipeparams_list = pipeparams_list, pipe_geometry = pipe_geometry)

#%%=======================================================================
# Sampling params
#=========================================================================
Ns = 150     # no of samples in each chain
Nb = 50       # Burnin
Nt = 1#50         # Thinning
sample_scale = 1e-3 # Initial sample scale

#%%=======================================================================
# Define model
#=========================================================================
# Scan Geometry parameters
DetectorCount = 300 # no of detectors
AngleCount = 100 # no of view angles
maxAngle = 2*np.pi
angles = np.linspace(0,maxAngle,AngleCount, endpoint=True)
source_object_dist = 7
object_detector_dist = 5
det_spacing = 10/DetectorCount
m = DetectorCount * AngleCount

# Model
A = ShiftedFanBeam2DModel(im_size = (N,N),
                    det_count = DetectorCount,
                    angles = angles,
                    source_y = -source_object_dist,
                    detector_y = object_detector_dist,
                    beamshift_x = -1.2,
                    det_spacing = 4/DetectorCount,
                    domain = (imagesize,imagesize))

# Configure model
A.domain_geometry = pipe_geometry

show_geometry(A.acquisition_geometry, A.image_geometry)
plt.savefig(resultpath + resultname + '_ag.png')

# FP = A(CUQIarray([0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.1, 0.2], geometry = pipe_geometry))

# fig, ax = plt.subplots(1,1, figsize=(5,3))
# cs = FP.plot(aspect = 1/2, extent = [0, 300, 360, 0], interpolation = "none")
# cs[0].axes.set_xticks(np.linspace(0,300, 5, endpoint = True))
# cs[0].axes.set_yticks(np.linspace(0,360, 7, endpoint = True))
# cs[0].axes.set_xlabel('Detector')
# cs[0].axes.set_ylabel('View angle [degree]')
# fig.subplots_adjust(right=0.85, bottom=0.15)
# cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
# cbar = plt.colorbar(cs[0], cax=cax)
# plt.savefig(resultpath + resultname +  '_FP.png')

# sys.exit()

#%%=======================================================================
# Synthetic data
#=========================================================================

pipeparams_list_phantom = [DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10, DF11]
pipe_geometry_phantom = DiskFree(nolayers, imagesize, N_phantom)

# Model
A_phantom = ShiftedFanBeam2DModel(im_size = (N_phantom,N_phantom),
                    det_count = DetectorCount,
                    angles = angles,
                    source_y = -source_object_dist,
                    detector_y = object_detector_dist,
                    beamshift_x = -1.2,
                    det_spacing = 4/DetectorCount,
                    domain = (imagesize,imagesize))

# Configure model
A_phantom.domain_geometry = pipe_geometry_phantom

# prior
PPCollection_phantom = PipeParamsCollection(pipeparams_list = pipeparams_list_phantom, pipe_geometry = pipe_geometry_phantom)
theta_phantom = PPCollection_phantom.get_prior()

# True values in CUQIarray
theta_true = PPCollection_phantom.get_truth()

# data 
noise_std = 0.1
d_phantom  = Gaussian(mean = A_phantom(theta_phantom), sqrtcov = noise_std, geometry=A_phantom.range_geometry)
np.random.seed(10)
d_obs = d_phantom(theta_phantom = theta_true).sample()

fig, ax = plt.subplots(1,1, figsize=(5,3))
cs = d_obs.plot(aspect = 1/2, extent = [0, 300, 360, 0], interpolation = "none")
cs[0].axes.set_xticks(np.linspace(0,300, 5, endpoint = True))
cs[0].axes.set_yticks(np.linspace(0,360, 7, endpoint = True))
cs[0].axes.set_xlabel('Detector')
cs[0].axes.set_ylabel('View angle [degree]')
fig.subplots_adjust(right=0.85, bottom=0.15)
cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
cbar = plt.colorbar(cs[0], cax=cax)
plt.savefig(resultpath + resultname +  '_sinogram.png')

#%%=======================================================================
# Specification and sampling of Bayesian problem
#=========================================================================

# prior
cx0 = DF1.prior
cy0 = DF2.prior
r0 = DF3.prior
cx1 = DF4.prior
cy1 = DF5.prior
r1 =  DF6.prior
phi1 = DF7.prior
cx2 = DF8.prior
cy2 = DF9.prior
r2 = DF10.prior
phi2 = DF11.prior

# data
d  = Gaussian(mean = lambda cx0, cx1, cx2, cy0, cy1, cy2, r0, r1, r2, phi1, phi2: A(CUQIarray([cx0, cx1, cx2, cy0, cy1, cy2, r0, r1, r2, phi1, phi2], geometry = pipe_geometry)), 
                sqrtcov = noise_std, geometry=A.range_geometry)
# cx0, cx1, cx2, cy0, cy1, cy2, r0, r1, r2, phi1, phi2
# 0, 0, 0, 0, 0, 0, 0.4, 0.9, 1.1, 0.7, 0.3

# posterior
posterior = JointDistribution(cx0, cx1, cx2, cy0, cy1, cy2, r0, r1, r2, phi1, phi2, d)(d=d_obs)

print(posterior(cx1=0, cx2=0, cy0=0, cy1=0, cy2=0, r0=0.4, r1=0.9, r2=1.1, phi1=0.7, phi2=0.3).logd(1))

# sample
np.random.seed(10)
# setup initial guess as random sample from prior
theta = PPCollection.get_prior()
theta0 = theta.sample(1)

# Gibbs sampler
sampling_strategy = {
    "cx0" : MHNew(scale = sample_scale, initial_point = 0),
    "cx1" : MHNew(scale = sample_scale, initial_point = 0),
    "cx2" : MHNew(scale = sample_scale, initial_point = 0),
    "cy0" : MHNew(scale = sample_scale, initial_point = 0),
    "cy1" : MHNew(scale = sample_scale, initial_point = 0),
    "cy2" : MHNew(scale = sample_scale, initial_point = 0),
    "r0" : MHNew(scale = sample_scale, initial_point = 0.4),
    "r1" : MHNew(scale = sample_scale, initial_point = 0.9),
    "r2" : MHNew(scale = sample_scale, initial_point = 1.1),
    "phi1" : MHNew(scale = sample_scale, initial_point = 0.7),
    "phi2" : MHNew(scale = sample_scale, initial_point = 0.3)
}

sampler = HybridGibbsNew(posterior, sampling_strategy)

print(sampler)

# warmup
sampler.warmup(Nb)
# sample
sampler.sample(Ns)
samples_all = sampler.get_samples()
samples = samples_all.burnthin(Nb)

#%%=======================================================================
# save data
#=========================================================================

paramnames = pipe_geometry.variables

# plot chain
plt.figure()
samples_all.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
plt.savefig(resultpath + resultname + '_chainswithwarmup.png')

#%%=======================================================================
# save data
#=========================================================================
with open('{}{}.pkl'.format(resultpath,resultname), 'wb') as f:  # Python 3: open(..., 'wb')
    dill.dump([samples, sampler, PPCollection, A], f)