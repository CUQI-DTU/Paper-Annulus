###################################################
# Main script for running experiments with geometric parameterization of pipes
# By Silja L. Christensen
# June 2024
###################################################

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# import dill
from typing import List, Union
from copy import copy, deepcopy

sys.path.append("../CUQIpy") 
sys.path.append("../CUQIpy-CIL") 

from AnnulusGeometry2024 import PipeParam, PipeParamsCollection, DiskFree, DiskConcentric, AnnulusFree, AnnulusConcentricConnected
# cuqipy version 1.0.0
import cuqi
from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, Gamma, InverseGamma, Uniform, JointDistribution, Posterior
from cuqi.geometry import Image2D
from cuqi.sampler import CWMH, Gibbs
from cuqi.experimental.mcmc import Sampler, CWMH, HybridGibbs, LinearRTO, Direct
from cuqi.likelihood import Likelihood
from cuqi.model import Model, AffineModel
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

#%% funs

def check(p1, p2, base_array):
    """
    Source: https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array


class myIGConjugate(Sampler):
    def __init__(self, target = None, s_minval = 1e-7):
        super().__init__(target)
        
        self.s_minval = s_minval
    
    def _initialize(self):
        pass

    def validate_target(self):
        pass

    def tune(self, skip_len, update_count):
        pass

    def step(self, x=None):
        # Extract variables
        b = self.target.get_density("d").data   #get_density    #d
        alpha = self.target.get_density("s").shape                                 #alpha
        beta = self.target.get_density("s").scale                                  #beta

        # Conjugate Inverse Gamma distribution and sample it
        samp = InverseGamma(shape=1/2+alpha,location=0,scale=.5*b**2+beta).sample()
        self.current_point = np.maximum(self.s_minval, samp)
        acc = 1
        return acc

class myGammaSampler(Sampler):
    def __init__(self, target = None):
        self.target = target
        super().__init__(target)
    
    def _initialize(self):
        pass

    def validate_target(self):
        pass

    def tune(self, skip_len, update_count):
        pass

    def step(self, x=None):
        # Extract variables
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                #beta
        
        # Conjugate Inverse Gamma distribution and sample it
        self.current_point = np.random.gamma(shape=alpha,scale=1/beta)
        acc = 1
        return acc

class SumOfModels:
    """ A sum of models is defined by a list of models and represents the sum of the models.

    Consider a list of models [model_1(x), model_2(y), model_3(x), ...]. The sum model is defined as

    .. math::

        model_{sum}(x, y) = model_1(x) + model_2(y) + model_3(x) + ...

    As indicated, models with the same parameter can be used as part of the sum model.

    Parameters
    ----------
    models : Model
        The models to include in the sum model.
        Each model is passed as comma-separated arguments.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from cuqi import LinearModel, SumOfModels

        # Define two linear models
        model_1 = Model(lambda x: x, 1, 1)
        model_2 = Model(lambda y: y+1, 1, 1)

        # Define the sum of the two models
        # model_1 + model_2 also works
        sum_model = SumOfModels(model_1, model_2)

        # Evaluate the sum model
        sum_model(x=1, y=2) # Returns 4

        # Partial evaluation
        sum_model(x=1) # Returns a new model with model_1(x) fixed
        
    """
    def __init__(self, *models: Model):
        self._models = list(models)
        self._shift = 0 # Initial shift

    # Options for return is SumModel, Model or float
    def __call__(self, *args, **kwargs) -> Union[SumOfModels, Model, float]:
        """ Evaluate the model at the given parameters. """

        kwargs = self._parse_args_add_to_kwargs(*args, **kwargs)

        # Evaluation happens in a shallow copy of the SumModel
        new_model = copy(self)              # Shallow copy of self
        new_model._models = [copy(model) for model in self._models] # Shallow copy of models
        # Go through each keyword argument and each model
        for kwarg, value in kwargs.items():
            for model in new_model._models:        
                # Evaluate model if kwarg matches _non_default_args
                if kwarg in cuqi.utilities.get_non_default_args(model):

                    # make dict of kwarg and value and evaluate model
                    new_model._shift += model(**{kwarg: value})

                    # Remove model from list since it has been evaluated
                    new_model._models.remove(model)

        if len(new_model._models) == 0: # Model has been evaluated fully
            return new_model._shift

        if len(new_model._models) == 1: # Single model left, return it (including shift which is the other evaluated models)
            new_model._models[0]._shift = new_model._shift
            return new_model._models[0]

        return new_model # Else return the SumModel


    @property
    def range_dim(self):
        """ Return the range dimension of the sum model. """
        return self._models[0].range_dim
    
    @property
    def domain_dim(self):
        """ Return the domain dimension of the sum model. """

        # Extract domain dimensions of all models
        domain_dims = [model.domain_dim for model in self._models]

        # If only one parameter return domain dimension of model
        if len(self._non_default_args) == 1:
            if len(set(domain_dims)) > 1:
                raise ValueError("SumOfModels: Models with same parameter must have the same domain dimension.")
            return self._models[0].domain_dim
        else: #If more than one parameter, return list of domain dimensions
            return domain_dims

    @property
    def _non_default_args(self):
        """ Return non-default args of all models. """

        # Return non-default args of all models
        L = [cuqi.utilities.get_non_default_args(model) for model in self._models]

        # Make a single list
        single_L = [item for sublist in L for item in sublist]

        # Remove duplicates but keep order of elements
        return list(dict.fromkeys(single_L))
    
    def _parse_args_add_to_kwargs(self, *args, **kwargs):
        """ Private function that parses the input arguments of the model and adds them as keyword arguments matching the non default arguments of the forward function. """

        if len(args) > 0:

            if len(kwargs) > 0:
                raise ValueError("The model input is specified both as positional and keyword arguments. This is not supported.")
                
            if len(args) != len(self._non_default_args):
                raise ValueError("The number of positional arguments does not match the number of non-default arguments of the model.")
            
            # Add args to kwargs following the order of non_default_args
            for idx, arg in enumerate(args):
                kwargs[self._non_default_args[idx]] = arg

        return kwargs

    def __len__(self):
        return self._models[0].range_dim

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__} of {len(self._models)} models. Parameters: {self._non_default_args}. Models:\n"
        for model in self._models:
            msg += f"  {model}\n"
        if self._shift != 0:
            msg += f"Shift:  {self._shift}"
        return msg

 #%%=======================================================================
# Paths
#=========================================================================

# path for saving results
resultpath = '../../../../../../work3/swech/results/'
resultname = 'DecomposedTest_thetatrue'
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
# Pipe params
#=========================================================================

nolayers = 1

pipeparams_list = [DF1, DF2, DF3, DF4, DF5, DF6, DF7]#, DF8, DF9, DF10, DF11]

pipe_geometry = DiskFree(nolayers, imagesize, N)

# Collect the info above in one object
PPCollection = PipeParamsCollection(pipeparams_list = pipeparams_list, pipe_geometry = pipe_geometry)

#%%=======================================================================
# Sampling params
#=========================================================================
Ns = 5      # no of samples in each chain
Nb = 800       # Burnin
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
                    beamshift_x = 0,
                    det_spacing = det_spacing,
                    domain = (imagesize,imagesize))

show_geometry(A.acquisition_geometry, A.image_geometry)
plt.savefig(resultpath + resultname + '_ag.png')

#%%=======================================================================
# Synthetic data
#=========================================================================

pipeparams_list_phantom = [DF1, DF2, DF3, DF4, DF5, DF6, DF7]#, DF8, DF9, DF10, DF11]
pipe_geometry_phantom = DiskFree(nolayers, imagesize, N_phantom)

# Model
A_phantom = ShiftedFanBeam2DModel(im_size = (N_phantom,N_phantom),
                    det_count = DetectorCount,
                    angles = angles,
                    source_y = -source_object_dist,
                    detector_y = object_detector_dist,
                    beamshift_x = 0,
                    det_spacing = det_spacing,
                    domain = (imagesize,imagesize))

# prior
PPCollection_phantom = PipeParamsCollection(pipeparams_list = pipeparams_list_phantom, pipe_geometry = pipe_geometry_phantom)
theta_phantom = PPCollection_phantom.get_prior()

# True values in CUQIarray
theta_true = PPCollection_phantom.get_truth()

# Make phantom with defects
phantom = pipe_geometry_phantom.par2fun(theta_true)

# radial cracks
no = 2
c = np.round(N_phantom/2+np.array([-0.1/imagesize*N_phantom,0.2/imagesize*N_phantom]))
ang = np.array([np.pi/4, 3*np.pi/4])
dist = 0.65/imagesize*N_phantom
w = 0.04/imagesize*N_phantom*np.ones(no)
l = 0.3/imagesize*N_phantom*np.ones(no)
vals = np.array([1, 0]) # Positive, negative

defectmask = []

for i in range(no):
    # coordinates in (x,y), -1 to 1 system
    coordinates0 = np.array([
        [c[0]+w[i]/2, c[1]+dist + l[i]/2],
        [c[0]-w[i]/2, c[1]+dist + l[i]/2],
        [c[0]-w[i]/2, c[1]+dist - l[i]/2],
        [c[0]+w[i]/2, c[1]+dist - l[i]/2]
    ])
    R = np.array([
        [np.cos(ang[i]), -np.sin(ang[i])],
        [np.sin(ang[i]), np.cos(ang[i])]
        ])
    # Rotate around image center
    coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
    coordinates = coordinates.T

    # transform into (row, column) indicies
    vertices = np.ceil(np.fliplr(coordinates))
    # create mask
    tmpmask = create_polygon([N_phantom,N_phantom], vertices)
    defectmask.append(np.array(tmpmask, dtype=bool))
    phantom[defectmask[i]] = vals[i]

phantom = CUQIarray(phantom, is_par = False, geometry = A_phantom.domain_geometry) 

fig, ax = plt.subplots(1,1, figsize=(5,3))
phantom.plot()
plt.savefig(resultpath + resultname +  '_phantom.png')

# data 
noise_std = 0.01
np.random.seed(10)
y_obs  = Gaussian(mean = A_phantom(phantom), sqrtcov = noise_std, geometry=A_phantom.range_geometry).sample()

fig, ax = plt.subplots(1,1, figsize=(5,3))
cs = y_obs.plot(aspect = 1/2, extent = [0, 300, 360, 0], interpolation = "none")
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

# Pipe prior
theta = PPCollection.get_prior()

# Hierarchical defect prior, NOT TESTED
# defect prior
omega0 = 4  # turning up to 8 or 12 makes defects slightly narrower, but it becomes harder to reconstruct the very narrow defect
s_bc = 1e-6
s_init = 1e-6
flat_order = "F"
# Prior on d
def d_sqrtprec(s_):
    return np.sqrt(1/s_)
d = Gaussian(mean = np.zeros(N**2), sqrtprec = lambda s: d_sqrtprec(s), geometry=Image2D((N,N)))
d0 = CUQIarray(1e-3*np.ones(N**2), geometry=Image2D((N,N)))
# Hyperprior on s
def s_scale(w_):
    # Weights computed from auxilirary variables
    W = np.reshape(w_, (N+1, N+1), order = flat_order)
    w1 = W[:-1,:-1].flatten(order=flat_order) # w_(i,j)
    w2 = W[1:,:-1].flatten(order=flat_order) # w_(i+1,j)
    w3 = W[:-1,1:].flatten(order=flat_order) # w_(i,j+1)
    w4 = W[1:,1:].flatten(order=flat_order) # w_(i+1,j+1)
    omega1 = (w1+w2+w3+w4)/4
    return omega0*omega1
s = InverseGamma(shape=omega0, location = 0, scale=lambda w: s_scale(w), geometry=Image2D((N,N)))
s0 = CUQIarray(s_init*np.ones(N**2), geometry=Image2D((N,N)))
# Hyperprior on w
def w_rate(s_):
    # Weights computed from s
    # add row and column to S. Defines dirichlet boundary condition.
    S = s_bc*np.ones((N+2, N+2))
    S[1:-1,1:-1] = np.reshape(s_, (N, N), order = flat_order)
    s1 = S[1:,1:].flatten(order=flat_order) # s_(i,j)
    s2 = S[:-1,1:].flatten(order=flat_order) # s_(i-1,j)
    s3 = S[1:,:-1].flatten(order=flat_order) # s_(i,j-1)
    s4 = S[:-1,:-1].flatten(order=flat_order) # s_(i-1,j-1)
    omega2 = (1/s1 + 1/s2 + 1/s3 + 1/s4)/4
    return (omega0*omega2) # 1/(alpha*omega2)
w = Gamma(shape=omega0, rate=lambda s: w_rate(s), geometry=Image2D((N+1,N+1)))
w0 = CUQIarray(s_init*np.ones((N+1)**2), geometry=Image2D((N+1,N+1)))

# Configure model
A.domain_geometry = pipe_geometry
A_theta = copy(A)(theta)
A.domain_geometry = Image2D((N,N))
A_d = copy(A)(d)
A_joint = SumOfModels(A_theta, A_d)

# print(np.max(A_joint(theta = theta_true)._shift))

# print(np.max(A_joint._shift))
# print(np.max(A_d._shift))

# sampling, phase 1, CWMH
# y1 = Gaussian(mean = A_theta, sqrtcov = noise_std, geometry=A.range_geometry)
# posterior1 = JointDistribution(theta, y1)(y1=y_obs)
# np.random.seed(10)
# theta0 = theta.sample(1)
# theta0[0:4] = 0
# sampler_burnin = CWMH(posterior1, scale = sample_scale, initial_point = theta0)
# sampler_burnin.warmup(Nb)
# plt.figure()
# sampler_burnin.get_samples().plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
# plt.savefig(resultpath + resultname + '_thetachainswarmup.png')

# sampling phase 2, Gibbs
# theta_state = sampler_burnin.get_state()
# theta.init_point = theta_state['state']['current_point'] # array?
theta.init_point = theta_true
d.init_point = d0
s.init_point = s0
w.init_point = w0

y2 = Gaussian(mean = A_joint, sqrtcov = noise_std, geometry=A.range_geometry)
posterior2 = JointDistribution(theta, d, s, w, y2)(y2=y_obs)

#print(A_d.gradient(direction = y_obs, wrt = d0))


sys.exit()

sampling_strategy = {
    "theta" : CWMH(scale = 1e-3),#theta_state['state']['scale']),
    "d" : LinearRTO(),
    "s" : myIGConjugate(),
    "w" : myGammaSampler()
}

sampler_Gibbs = HybridGibbs(posterior2, sampling_strategy)
sampler_Gibbs.sample(Ns)
samplesGibbs = sampler_Gibbs.get_samples()

#%%=======================================================================
# save data
#=========================================================================

# plot chain
# plt.figure()
# samplesGibbs["theta"].plot_chain(variable_indices=range(pipe_geometry.par_shape[0]))
# plt.savefig(resultpath + resultname + '_thetachains.png')

plt.figure()
samplesGibbs["d"].plot_chain()
plt.savefig(resultpath + resultname + '_dchains.png')

plt.figure()
samplesGibbs["d"].plot_mean()
plt.savefig(resultpath + resultname + '_dmean.png')

plt.figure()
samplesGibbs["d"].plot_std()
plt.savefig(resultpath + resultname + '_dstd.png')

plt.figure()
samplesGibbs["s"].plot_chain()
plt.savefig(resultpath + resultname + '_schains.png')

plt.figure()
samplesGibbs["s"].plot_mean()
plt.savefig(resultpath + resultname + '_smean.png')

plt.figure()
samplesGibbs["s"].plot_std()
plt.savefig(resultpath + resultname + '_sstd.png')

plt.figure()
samplesGibbs["w"].plot_chain()
plt.savefig(resultpath + resultname + '_wchains.png')

plt.figure()
samplesGibbs["w"].plot_mean()
plt.savefig(resultpath + resultname + '_wmean.png')

plt.figure()
samplesGibbs["w"].plot_std()
plt.savefig(resultpath + resultname + '_wstd.png')

#%%=======================================================================
# save data
#=========================================================================
# with open('{}{}.pkl'.format(resultpath,resultname), 'wb') as f:  # Python 3: open(..., 'wb')
#     dill.dump([sampler1, PPCollection, phantom, A], f)