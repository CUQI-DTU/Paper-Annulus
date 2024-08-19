###################################################
# Main script for running experiments with geometric parameterization of pipes
# By Silja L. Christensen
# June 2024
###################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import dill
import sys

from AnnulusGeometry2024 import PipeParam, PipeParamsCollection, DiskFree, DiskConcentric, AnnulusFree, AnnulusConcentricConnected
# cuqipy version 1.0.0
from cuqi.distribution import Gaussian, Gamma, Uniform, JointDistribution
from cuqi.sampler import CWMH
from cuqi.samples import Samples
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

# AnnulusCC
ACC1 = PipeParam(paramtype = "center_x", 
                layerno = 1,
                truevalue = 0.1,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
ACC2 = PipeParam(paramtype = "center_y", 
                layerno = 1,
                truevalue = 0.2,
                prior=Gaussian(mean = 0, sqrtcov = 0.5))
ACC3 = PipeParam(paramtype = "radius", 
                layerno = 1, 
                truevalue = 0.4,
                prior=Uniform(low = 0.3, high = 0.5))
ACC4 = PipeParam(paramtype = "width", 
                layerno = 1,
                truevalue = 0.5,
                prior=Uniform(low = 0.4, high = 0.6))
ACC5 = PipeParam(paramtype = "abscoeff", 
                layerno = 1,
                truevalue = 0.7,
                prior=Gamma(shape = 2, rate = 2))

#%%=======================================================================
# Parameter lib
#=========================================================================

nolayers = 1

pipeparams_list = [ACC1, ACC2, ACC3, ACC4, ACC5]

pipe_geometry = AnnulusConcentricConnected(nolayers, imagesize, N)

# Collect the info above in one object
PPCollection = PipeParamsCollection(pipeparams_list = pipeparams_list, pipe_geometry = pipe_geometry)

#%%=======================================================================
# Sampling params
#=========================================================================
Ns = 2000     # no of samples in each chain
Nb = 4000       # Burnin
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
                    beamshift_x = 0,#-1.2,
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

pipeparams_list_phantom = [ACC1, ACC2, ACC3, ACC4, ACC5]
pipe_geometry_phantom = AnnulusConcentricConnected(nolayers, imagesize, N_phantom)

# Model
A_phantom = ShiftedFanBeam2DModel(im_size = (N_phantom,N_phantom),
                    det_count = DetectorCount,
                    angles = angles,
                    source_y = -source_object_dist,
                    detector_y = object_detector_dist,
                    beamshift_x = 0,#-1.2,
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
# Specification of prior, data distribution and posterior
#=========================================================================

# cx = ACC1.prior
# cy = ACC2.prior
# r = ACC3.prior
# w = ACC4.prior
# phi = ACC5.prior

# # prior
# theta = PPCollection.get_prior()

# # data 
# d  = Gaussian(mean = A(theta), sqrtcov = noise_std, geometry=A.range_geometry)

# # posterior
# posterior = JointDistribution(theta, d)(d=d_obs)

# # data
# d  = Gaussian(mean = lambda cx, cy, r, w, phi: A(np.array([cx, cy, r, w, phi])), 
#                 sqrtcov = noise_std, geometry=A.range_geometry)

# # posterior
# posterior = JointDistribution(cx, cy, r, w, phi, d)(d=d_obs)

#%%=======================================================================
# CWMH vs Gibbs to illustrate sampling scale problem
#=========================================================================

################### CWMH ########################
# prior
theta = PPCollection.get_prior()
# data 
d  = Gaussian(mean = A(theta), sqrtcov = noise_std, geometry=A.range_geometry)
# posterior
posterior = JointDistribution(theta, d)(d=d_obs)

np.random.seed(10)
# New CWMH
samplerCWMH = CWMHNew(posterior, scale = sample_scale)
# warmup
#samplerCWMH.warmup(Nb)
# sample
samplerCWMH.sample(Nb+Ns)
samplesCWMH = samplerCWMH.get_samples()

plt.figure()
samplesCWMH.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
plt.savefig(resultpath + resultname + '_allchainsCWMH.png')
for i in range(pipe_geometry.par_shape[0]):
    plt.figure()
    samplesCWMH.burnthin(Nb).plot_chain(variable_indices=i)
    plt.savefig(resultpath + resultname + '_chain{}CWMH.png'.format(i))

################### Gibbs #######################
# prior
cx = ACC1.prior
cy = ACC2.prior
r = ACC3.prior
w = ACC4.prior
phi = ACC5.prior
# data
d  = Gaussian(mean = lambda cx, cy, r, w, phi: A(np.array([cx, cy, r, w, phi])), 
                sqrtcov = noise_std, geometry=A.range_geometry)
# posterior
posterior = JointDistribution(cx, cy, r, w, phi, d)(d=d_obs)

np.random.seed(10)
# Gibbs sampler
sampling_strategy = {
    "cx" : MHNew(scale = sample_scale),
    "cy" : MHNew(scale = sample_scale),
    "r" : MHNew(scale = sample_scale),
    "w" : MHNew(scale = sample_scale),
    "phi" : MHNew(scale = sample_scale)
}

samplerGibbs = HybridGibbsNew(posterior, sampling_strategy)

# warmup
#samplerGibbs.warmup(Nb)
# sample
samplerGibbs.sample(Nb+Ns)
samplesGibbs = samplerGibbs.get_samples()

samples_array = np.array([samplesGibbs[key].samples for key in samplesGibbs.keys()]).reshape(len(samplesGibbs.keys()), -1)
samplesGibbs = Samples(samples_array, geometry = pipe_geometry)

plt.figure()
samplesGibbs.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
plt.savefig(resultpath + resultname + '_allchainsGibbs.png')
for i in range(pipe_geometry.par_shape[0]):
    plt.figure()
    samplesGibbs.burnthin(Nb).plot_chain(variable_indices=i)
    plt.savefig(resultpath + resultname + '_chain{}Gibbs.png'.format(i))


#%%=======================================================================
# Illustrattion of problem with initial points in Gibbs
#=========================================================================
# prior
cx = ACC1.prior
cy = ACC2.prior
r = ACC3.prior
w = ACC4.prior
phi = ACC5.prior
# data
d  = Gaussian(mean = lambda cx, cy, r, w, phi: A(np.array([cx, cy, r, w, phi])), 
                sqrtcov = noise_std, geometry=A.range_geometry)
# posterior
posterior = JointDistribution(cx, cy, r, w, phi, d)(d=d_obs)

np.random.seed(10)

# Gibbs sampler
sampling_strategy = {
    "cx" : MHNew(scale = sample_scale),
    "cy" : MHNew(scale = sample_scale),
    "r" : MHNew(scale = sample_scale),
    "w" : MHNew(scale = sample_scale),
    "phi" : MHNew(scale = sample_scale)
}

# Set initial points in distributions (not in sampling strategy) - Interface should be improved
cx.init_point = 0
cy.init_point = 0
r.init_point = 0.3
w.init_point = 0.4
phi.init_point = 0.6

# Gives error if posterior is placed here after init point is sat as above
#posterior = JointDistribution(cx, cy, r, w, phi, d)(d=d_obs) 

samplerInitPoint = HybridGibbsNew(posterior, sampling_strategy)

# warmup
samplerInitPoint.warmup(Nb)
# sample
samplerInitPoint.sample(Ns)
samplesInitPoint = samplerInitPoint.get_samples()
samples_array = np.array([samplesInitPoint[key].samples for key in samplesInitPoint.keys()]).reshape(len(samplesInitPoint.keys()), -1)
samplesInitPoint = Samples(samples_array, geometry = pipe_geometry)

plt.figure()
samplesInitPoint.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
plt.savefig(resultpath + resultname + '_allchainsInitPoint.png')


#%%=======================================================================
# save data
#=========================================================================
# with open('{}{}.pkl'.format(resultpath,resultname), 'wb') as f:  # Python 3: open(..., 'wb')
#     dill.dump([samples, sampler, PPCollection, A], f)
