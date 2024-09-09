###################################################
# Main script for running experiments with geometric parameterization of pipes
# By Silja L. Christensen
# June 2024
###################################################
# %%
import numpy as np
import matplotlib.pyplot as plt
import os
#import dill
import sys

from AnnulusGeometry2024 import PipeParam, PipeParamsCollection, DiskFree, DiskConcentric, AnnulusFree, AnnulusConcentricConnected
# cuqipy version 1.0.0
from cuqi.distribution import Gaussian, Gamma, Uniform, JointDistribution, Distribution
from cuqi.samples import Samples
from cuqi.experimental.mcmc import CWMH, HybridGibbs, MH, ProposalBasedSampler
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

#%%
class myMH(ProposalBasedSampler): # copyed from CUQIpy, but this way I can easily test small changes
    """ Metropolis-Hastings (MH) sampler.

    Parameters
    ----------
    target : cuqi.density.Density
        Target density or distribution.

    proposal : cuqi.distribution.Distribution or callable
        Proposal distribution. If None, a random walk MH is used (i.e., Gaussian proposal with identity covariance).

    scale : float
        Scaling parameter for the proposal distribution.

    kwargs : dict
        Additional keyword arguments to be passed to the base class :class:`ProposalBasedSampler`.

    """

    _STATE_KEYS = ProposalBasedSampler._STATE_KEYS.union({'scale', '_scale_temp'})

    def __init__(self, target=None, proposal=None, scale=1, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale, **kwargs)

    def _initialize(self):
        # Due to a bug? in old MH, we must keep track of this extra variable to match behavior.
        self._scale_temp = self.scale

    def validate_target(self):
        # Fail only when there is no log density, which is currently assumed to be the case in case NaN is returned.
        if np.isnan(self.target.logd(self._default_initial_point)):
            raise ValueError("Target does not have valid logd")

    def validate_proposal(self):
        if not isinstance(self.proposal, Distribution):
            raise ValueError("Proposal must be a cuqi.distribution.Distribution object")
        if not self.proposal.is_symmetric:
            raise ValueError("Proposal must be symmetric")

    def step(self):
        # propose state
        xi = self.proposal.sample(1)   # sample from the proposal
        x_star = self.current_point + self.scale*xi.flatten()   # MH proposal

        # evaluate target
        target_eval_star = self.target.logd(x_star)

        # ratio and acceptance probability
        ratio = target_eval_star - self.current_target_logd # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_point = x_star
            self.current_target_logd = target_eval_star
            acc = 1
        
        return acc

    def tune(self, skip_len, update_count):
        
        print(len(self._acc))
        hat_acc = np.mean(self._acc[-skip_len:])

        # d. compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes

        # We use self._scale_temp here instead of self.scale in update. This might be a bug,
        # but is equivalent to old MH
        self._scale_temp = np.exp(np.log(self._scale_temp) + zeta*(hat_acc-0.234))

        # update parameters
        self.scale = min(self._scale_temp, 1)

# %%
# Settings
save_fig = True

#%%=======================================================================
# Paths
#=========================================================================

# path for saving results
if save_fig:
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
                    beamshift_x = 0,
                    det_spacing = det_spacing,
                    domain = (imagesize,imagesize))

# Configure model
A.domain_geometry = pipe_geometry

show_geometry(A.acquisition_geometry, A.image_geometry)
if save_fig:
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
                    beamshift_x = 0,
                    det_spacing = det_spacing,
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
if save_fig:
    plt.savefig(resultpath + resultname +  '_sinogram.png')

#%%=======================================================================
# CWMH vs Gibbs to illustrate sampling scale problem
#=========================================================================

Ns = 500    # no of samples in each chain
Nb = 500       # Burnin
Nt = 1#50         # Thinning
sample_scale = 1e-2 # Initial sample scale
theta0 = np.array([0, 0, 0.3, 0.4, 0.6]) # Initial point

################### CWMH ########################
# prior
theta = PPCollection.get_prior()
# data 
d  = Gaussian(mean = A(theta), sqrtcov = noise_std, geometry=A.range_geometry)
# posterior
posterior = JointDistribution(theta, d)(d=d_obs)

# Print logd of posterior at initial point
print('Posterior logd at np.array([0, 0, 0.3, 0.4, 0.6]) = {}'.format(posterior.logd(theta0)) )

np.random.seed(10)
# New CWMH
samplerCWMH = CWMH(posterior, scale = sample_scale, initial_point = theta0)

#warmup
samplerCWMH.warmup(Nb, tune_freq = 1)

print("CWMH scales after warmup")
print(samplerCWMH.scale)

# sample
samplerCWMH.sample(Ns)
samplesCWMH = samplerCWMH.get_samples()

plt.figure()
samplesCWMH.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))

if save_fig:
    plt.savefig(resultpath + resultname + '_allchainsCWMH.png')

# for i in range(pipe_geometry.par_shape[0]):

#     plt.figure()
#     samplesCWMH.burnthin(Nb).plot_chain(variable_indices=i)

#     if save_fig:
#         plt.savefig(resultpath + resultname + '_chain{}CWMH.png'.format(i))


################### Gibbs #######################
# prior
cx = ACC1.prior
cy = ACC2.prior
r = ACC3.prior
w = ACC4.prior
phi = ACC5.prior

# init point
# Set initial points in distributions (not in sampling strategy) - Interface should be improved
# Must be set before creating the joint distribution
# Also must be arrays it seems! ;(
cx.init_point = np.array([theta0[0]])
cy.init_point = np.array([theta0[1]])
r.init_point = np.array([theta0[2]])
w.init_point = np.array([theta0[3]])
phi.init_point = np.array([theta0[4]])

# data
d  = Gaussian(mean = lambda cx, cy, r, w, phi: A(np.array([cx, cy, r, w, phi])), 
                sqrtcov = noise_std, geometry=A.range_geometry)
# posterior
posterior = JointDistribution(cx, cy, r, w, phi, d)(d=d_obs)

# Print logd of posterior at initial point
print('Posterior logd at cx=0, cy=0, r=0.3, w=0.4, phi=0.6 = {}'.format(posterior.logd(theta0[0], theta0[1], theta0[2], theta0[3], theta0[4])) )

np.random.seed(10)
# Gibbs sampler
sampling_strategy = {
    "cx" : myMH(scale = sample_scale),
    "cy" : myMH(scale = sample_scale),
    "r" : myMH(scale = sample_scale),
    "w" : myMH(scale = sample_scale),
    "phi" : myMH(scale = sample_scale)
}

samplerGibbs = HybridGibbs(posterior, sampling_strategy)

print("Gibbs scales before warmup")
print(samplerGibbs.samplers["cx"].scale)
print(samplerGibbs.samplers["cy"].scale)
print(samplerGibbs.samplers["r"].scale)
print(samplerGibbs.samplers["w"].scale)
print(samplerGibbs.samplers["phi"].scale)

# warmup
samplerGibbs.warmup(Nb)

print("Gibbs scales after warmup")
print(samplerGibbs.samplers["cx"].scale)
print(samplerGibbs.samplers["cy"].scale)
print(samplerGibbs.samplers["r"].scale)
print(samplerGibbs.samplers["w"].scale)
print(samplerGibbs.samplers["phi"].scale)

# sample
samplerGibbs.sample(Ns)

samplesGibbs = samplerGibbs.get_samples()

samples_array = np.array([samplesGibbs[key].samples for key in samplesGibbs.keys()]).reshape(len(samplesGibbs.keys()), -1)
samplesGibbs = Samples(samples_array, geometry = pipe_geometry)

plt.figure()
samplesGibbs.plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
if save_fig:
    plt.savefig(resultpath + resultname + '_allchainsGibbs.png')

# for i in range(pipe_geometry.par_shape[0]):

#     plt.figure()
#     samplesGibbs.burnthin(Nb).plot_chain(variable_indices=i)

#     if save_fig:
#         plt.savefig(resultpath + resultname + '_chain{}Gibbs.png'.format(i))


#%%=======================================================================
# save data
#=========================================================================
# with open('{}{}.pkl'.format(resultpath,resultname), 'wb') as f:  # Python 3: open(..., 'wb')
#     dill.dump([samples, sampler, PPCollection, A], f)
