###################################################
# This code was used to produce the results in 
# paper-draft: "Bayesian reconstruction of geometric 
# parameters that characterize subsea pipes using CT 
# measurements" that was included in the PhD thesis
# of Silja L. Christensen
# April 2024
###################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import dill

from Thesis_AnnulusGeometry import ConcentricConstrained, Organizer
# cuqipy version 0.3.0
import cuqi
# cuqipy-cil version 0.6.0
from cuqipy_cil.model import FanBeam2DModel
# CIL version 22.1
from cil.utilities.display import show_geometry

import subprocess
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

#%%=======================================================================
# Choises
#=========================================================================

# discretization 
N = 1024

# pipe prior
pipe_parameterization = "ConcentricConstrained"
nolayers = 1

# likelihood
noise_std = 0.1 

datapath = './synthdata/'
os.makedirs(datapath, exist_ok=True)
dataname = 'OneAnnulus_CC_std01'

np.random.seed(10)

#%%=======================================================================
# Define model
#=========================================================================
# Scan Geometry parameters
DetectorCount = 300 # no of detectors
domain = 4
AngleCount = 100 # no of view angles
maxAngle = 2*np.pi
theta = np.linspace(0,maxAngle,AngleCount, endpoint=True)
# AngleCount = 2
# theta = np.array([0,np.pi/2])
source_object_dist = 7
object_detector_dist = 5
det_spacing = 10/DetectorCount
m = DetectorCount * AngleCount

# Geometries
pipe_geom = ConcentricConstrained(nolayers=nolayers, imagesize=domain, pixeldim = N, c_coords='cartesian')

# Model
A = FanBeam2DModel(im_size = (N,N),
                                det_count = DetectorCount,
                                angles = theta,
                                source_object_dist = source_object_dist,
                                object_detector_dist = object_detector_dist,
                                det_spacing = det_spacing,
                                domain = (domain,domain))
# Configure model
A.domain_geometry = pipe_geom

#%%=======================================================================
# Pipe truth 
#=========================================================================

# param_organizer contains classes for each typer of parameter; Annulus parameters and defect parameters, maybe
Org = Organizer(pipe_geom)

if nolayers == 1:
    Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = -0.1, high = 0.3))
    Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = 0, high = 0.4))
    Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.5))
    Org.define_pipe_param("width", 0, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))
    Org.define_pipe_param("abscoeff", 0, random = True, value = 0.7, prior=cuqi.distribution.Uniform(low = 0.6, high = 0.8))
elif nolayers == 4:
    Org.define_pipe_param("center_x", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.6))
    Org.define_pipe_param("center_y", 0, random = True, value = -0.3, prior=cuqi.distribution.Uniform(low = -0.5, high = -0.1))
    Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.5))
    Org.define_pipe_param("width", 0, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = 0, high = 0.2))
    Org.define_pipe_param("abscoeff", 0, random = True, value = 0.7, prior=cuqi.distribution.Uniform(low = 0.6, high = 0.8))
    Org.define_pipe_param("width", 1, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))
    Org.define_pipe_param("abscoeff", 1, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = 0.1, high = 0.3))    
    Org.define_pipe_param("width", 2, random = True, value = 0.3, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.4))
    Org.define_pipe_param("abscoeff", 2, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))    
    Org.define_pipe_param("width", 3, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = 0.1, high = 0.3))
    Org.define_pipe_param("abscoeff", 3, random = True, value = 0.3, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.4))

# ground_truth
theta_true = Org.get_truth()
# Prior
theta = Org.get_prior()

#%%=======================================================================
# Generate noisy data
#=========================================================================
d  = cuqi.distribution.Gaussian(mean = A(theta), sqrtcov = noise_std, geometry=A.range_geometry)
np.random.seed(10)
d_obs = d(theta=theta_true).sample()

# %%=======================================================================
# save data
# =========================================================================

with open('{}{}.pkl'.format(datapath,dataname), 'wb') as f:  # Python 3: open(..., 'wb')
    dill.dump([theta_true, d_obs, noise_std], f)

# #%%=======================================================================
# # process and plot samples
# #=========================================================================

# # colorbar lims when plotting
cmin = -0.2
cmax = 1

cmap_image = "gray"

# Figure 2(b)/Figure 2(d): sinograms
fig, ax = plt.subplots(1,1, figsize=(5,3))
cs = d_obs.plot(aspect = 1/2, extent = [0, 300, 360, 0], interpolation = "none", cmap = cmap_image)
cs[0].axes.set_xticks(np.linspace(0,300, 5, endpoint = True))
cs[0].axes.set_yticks(np.linspace(0,360, 7, endpoint = True))
cs[0].axes.set_xlabel('Detector')
cs[0].axes.set_ylabel('View angle [degree]')
fig.subplots_adjust(right=0.85, bottom=0.15)
cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
cbar = plt.colorbar(cs[0], cax=cax)
plt.savefig(datapath + dataname +  '_sinogram.png')

# Figure 2(a)/Figure 2(c): phantoms
fig, ax = plt.subplots(1,1, figsize=(4,3))
cs = theta_true.plot(origin = "lower", interpolation = "none", extent = [-domain/2, domain/2, -domain/2, domain/2], vmin = cmin, vmax = cmax, cmap = cmap_image)
cs.axes.set_xticks(np.linspace(-domain/2,domain/2, 5, endpoint = True))
cs.axes.set_yticks(np.linspace(-domain/2,domain/2, 5, endpoint = True))
fig.subplots_adjust(right=0.95)
cax = fig.add_axes([cs.axes.get_position().x1+0.01,cs.axes.get_position().y0,0.03,cs.axes.get_position().height])
cbar = plt.colorbar(cs, cax=cax)
plt.savefig(datapath + dataname +  '_ground_truth_image.png')

# %%=======================================================================
# plot acquisition geometry
# =========================================================================
# Figure 3
show_geometry(A.acquisition_geometry, A.image_geometry)
plt.savefig(datapath + dataname + '_ag.png')