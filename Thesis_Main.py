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

from Thesis_AnnulusGeometry import ConcentricConnected, Free, Organizer
# cuqipy version 0.3.0
import cuqi
# cuqipy-cil version 0.6.0
from cuqipy_cil.model import FanBeam2DModel

import subprocess
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

#%%=======================================================================
# Choises
#=========================================================================

# path for saving results
resultpath = '../../../../../../work3/swech/results/'
resultname = 'OneAnnulus_CC_std01_Mixed'
os.makedirs(resultpath, exist_ok=True)

# path for loading data creates with script Thesis_GenerateSynthData.py
datapath = './synthdata/'
dataname = 'OneAnnulus_CC_std01'

# discretization 
N = 500

# pipe prior
pipe_parameterization = "ConcentricConnected"
nolayers = 1
prior_type = "Mixed" # For 1 annulus case: "Uniform" or "Mixed"

# sampling
Ns = 10000      # no of samples in each chain
Nb = 0       # Burnin
Nt = 1#50         # Thinning
sample_scale = 1e-3 # Initial sample scale
no_starts = 10
sample_likelihood = False

#%%=======================================================================
# Load data
#=========================================================================

with open('{}{}.pkl'.format(datapath, dataname), 'rb') as f:  # Python 3: open(..., 'rb')
    theta_true, d_obs, noise_std = dill.load(f)

#%%=======================================================================
# Define model
#=========================================================================
# Scan Geometry parameters
DetectorCount = 300 # no of detectors
domain = 4
AngleCount = 100 # no of view angles
maxAngle = 2*np.pi
theta = np.linspace(0,maxAngle,AngleCount, endpoint=True)
# Uncomment below for results with only 2 view angles
# AngleCount = 2
# theta = np.array([0,np.pi/2])
source_object_dist = 7
object_detector_dist = 5
det_spacing = 10/DetectorCount
m = DetectorCount * AngleCount

# Geometries
pipe_geom = ConcentricConnected(nolayers=nolayers, imagesize=domain, pixeldim = N, c_coords='cartesian')

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
# Pipe truth and prior
#=========================================================================

# Organize the pipe info
Org = Organizer(pipe_geom)

if nolayers == 1:
    if sample_likelihood == False:
        if prior_type == "Uniform":
            Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = -0.1, high = 0.3))
            Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = 0, high = 0.4))
            Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.5))
            Org.define_pipe_param("width", 0, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))
            Org.define_pipe_param("abscoeff", 0, random = True, value = 0.7, prior=cuqi.distribution.Uniform(low = 0.6, high = 0.8))
        elif prior_type == "Mixed":
            Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Gaussian(mean = 0, sqrtcov = 0.5))
            Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Gaussian(mean = 0, sqrtcov = 0.5))
            Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.5))
            Org.define_pipe_param("width", 0, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))
            Org.define_pipe_param("abscoeff", 0, random = True, value = 0.7, prior=cuqi.distribution.Gamma(shape = 2, rate = 2))
    else:
        Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = -1.5, high = 1.5))
        Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = -1.5, high = 1.5))
        Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0, high = 1))
        Org.define_pipe_param("width", 0, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0, high = 1))
        Org.define_pipe_param("abscoeff", 0, random = True, value = 0.7, prior=cuqi.distribution.Uniform(low = 0, high = 1.2))
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
# Sampling
#=========================================================================

# data 
d  = cuqi.distribution.Gaussian(mean = A(theta), sqrtcov = noise_std, geometry=A.range_geometry)
likelihood = cuqi.likelihood.Likelihood(d, d_obs)

# posterior
posterior = cuqi.distribution.JointDistribution(theta, d)(d=d_obs)

if sample_likelihood == False:
    target = posterior
else:
    target = likelihood.logd

# sample with multistart
np.random.seed(10)
samples = {}    # initiate dict to contain samples
for i in range(no_starts):
    # setup initial guess as random sample from prior
    theta0 = theta.sample(1)
    # setup CUQI sampler
    sampler = cuqi.sampler.CWMH(target, scale = sample_scale, x0 = theta0)
    # run sampler
    samples["start{0}".format(i)] = sampler.sample_adapt(N = Ns, Nb = Nb)

#%%=======================================================================
# save data
#=========================================================================
with open('{}{}.pkl'.format(resultpath,resultname), 'wb') as f:  # Python 3: open(..., 'wb')
    dill.dump([samples, Org, A], f)