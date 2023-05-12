#%%
import numpy as np
import matplotlib.pyplot as plt
import os
#import dill

from annulus_geometry2023 import ConcentricConstrained, Free, Organizer
# GitHub cuqi Spring 2023
import cuqi

from funs import FanBeam2DModel

import subprocess
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

# %load_ext autoreload
# %autoreload 2


#%%=======================================================================
# Choises
#=========================================================================

resultpath = './output/test/'
os.makedirs(resultpath, exist_ok=True)

# pipe prior
pipe_parameterization = "Free"
nolayers = 2

# defect prior
omega0 = 4  # turning up to 8 or 12 makes defects slightly narrower, but it becomes harder to reconstruct the very narrow defect
s_bc = 1e-6
s_init = 1e-5

# likelihood
rnl = 0.05 

# sampling
Ns = 6000      # no of samples in each chain
Nb = 1000       # Burnin
Nt = 50         # Thinning
sample_scale = 1e-3 # Initial sample scale


#%%=======================================================================
# Define model
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
N = 500

# Model
A = FanBeam2DModel(im_size = (N,N),
                                det_count = DetectorCount,
                                angles = theta,
                                source_object_dist = source_object_dist,
                                object_detector_dist = object_detector_dist,
                                det_spacing = det_spacing,
                                domain = (domain,domain),
                                proj_type = "cuda")

# Geometries
pipe_geom = Free(nolayers=nolayers, imagesize=domain, pixeldim = N, c_coords='cartesian')
defect_geom = A.domain_geometry

#%%=======================================================================
# Pipe prior
#=========================================================================

# param_organizer contains classes for each typer of parameter; Annulus parameters and defect parameters, maybe
Org = Organizer(pipe_geom)

Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Gaussian(0.1, sqrtcov = 0.05))
Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Gaussian(0.2, sqrtcov = 0.05))
Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Gaussian(0.4, sqrtcov = 0.05))
Org.define_pipe_param("abscoeff", 0, random = True, value = 0, prior=cuqi.distribution.Gaussian(0, sqrtcov = 0.01))
Org.define_pipe_param("center_x", 1, random = True, value = 0.1, prior=cuqi.distribution.Gaussian(0.1, sqrtcov = 0.05))
Org.define_pipe_param("center_y", 1, random = True, value = 0.2, prior=cuqi.distribution.Gaussian(0.2, sqrtcov = 0.05))
Org.define_pipe_param("radius", 1, random = True, value = 0.5, prior=cuqi.distribution.Gaussian(0.5, sqrtcov = 0.05))
Org.define_pipe_param("abscoeff", 1, random = True, value = 0.5, prior=cuqi.distribution.Gaussian(0.5, sqrtcov = 0.05))
Org.define_pipe_param("center_x", 2, random = True, value = 0.15, prior=cuqi.distribution.Gaussian(0.1, sqrtcov = 0.05))
Org.define_pipe_param("center_y", 2, random = True, value = 0.25, prior=cuqi.distribution.Gaussian(0.2, sqrtcov = 0.05))
Org.define_pipe_param("radius", 2, random = True, value = 1, prior=cuqi.distribution.Gaussian(1, sqrtcov = 0.05))
Org.define_pipe_param("abscoeff", 2, random = True, value = 0.7, prior=cuqi.distribution.Gaussian(0.7, sqrtcov = 0.05))

# Org.define_pipe_param("center_x", 0, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("center_y", 0, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("radius", 0, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.45))
# Org.define_pipe_param("abscoeff", 0, random = True, value = 0, prior=cuqi.distribution.Uniform(low = 0, high = 0.05))
# Org.define_pipe_param("center_x", 1, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("center_y", 1, random = True, value = 0.2, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("radius", 1, random = True, value = 0.1, prior=cuqi.distribution.Uniform(low = 0.05, high = 0.2))
# Org.define_pipe_param("abscoeff", 1, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.3, high = 0.7))
# Org.define_pipe_param("center_x", 2, random = True, value = 0.15, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("center_y", 2, random = True, value = 0.25, prior=cuqi.distribution.Uniform(low = -0.5, high = 0.5))
# Org.define_pipe_param("radius", 2, random = True, value = 0.5, prior=cuqi.distribution.Uniform(low = 0.4, high = 0.6))
# Org.define_pipe_param("abscoeff", 2, random = True, value = 0.7, prior=cuqi.distribution.Uniform(low = 0.5, high = 0.9))
# Org.define_pipe_param("radius", 3, random = True, value = 1.3,prior=cuqi.distribution.Uniform(low = 1.2, high = 1.4))
# Org.define_pipe_param("abscoeff", 3, random = True, value = 0.4, prior=cuqi.distribution.Uniform(low = 0.2, high = 0.6))
# Org.define_pipe_param("radius", 4, random = True, value = 1.5,prior=cuqi.distribution.Uniform(low = 1.41, high = 1.6))
# Org.define_pipe_param("abscoeff", 4, random = True, value = 0.8, prior=cuqi.distribution.Uniform(low = 0.6, high = 1))

# Prior
z = Org.get_prior(name = "z")
z_true = Org.get_truth()

# Configure model
A_z = A
A_z.domain_geometry = pipe_geom
A_z = A_z(z)
#%%=======================================================================
# Hierarchical defect prior, NOT TESTED
#=========================================================================

# if defect_recon == True:
#     # Prior on d
#     def d_sqrtprec(s_):
#         return np.sqrt(1/s_)
#     d = cuqi.distribution.Gaussian(mean = np.zeros(N**2), sqrtprec = lambda s: d_sqrtprec(s), geometry=defect_geom, name = "d")
#     d.init_point = np.zeros(N**2)
#     # Configure model
#     A_d = A(d)
#     A_d.domain_geometry = defect_geom

#     flat_order = "F"

#     # Hyperprior on s
#     def s_scale(w_):
#         # Weights computed from auxilirary variables
#         W = np.reshape(w_, (N+1, N+1), order = flat_order)
#         w1 = W[:-1,:-1].flatten(order=flat_order) # w_(i,j)
#         w2 = W[1:,:-1].flatten(order=flat_order) # w_(i+1,j)
#         w3 = W[:-1,1:].flatten(order=flat_order) # w_(i,j+1)
#         w4 = W[1:,1:].flatten(order=flat_order) # w_(i+1,j+1)
#         omega1 = (w1+w2+w3+w4)/4
#         return omega0*omega1
#     s = cuqi.distribution.InverseGamma(shape=omega0, location = 0, scale=lambda w: s_scale(w), geometry=defect_geom, name="s")
#     s.init_point = s_init*np.ones(N**2)

#     # Hyperprior on w
#     def w_rate(s_):
#         # Weights computed from s
#         # add row and column to S. Defines dirichlet boundary condition.
#         S = s_bc*np.ones((N+2, N+2))
#         S[1:-1,1:-1] = np.reshape(s_, (N, N), order = flat_order)
#         s1 = S[1:,1:].flatten(order=flat_order) # s_(i,j)
#         s2 = S[:-1,1:].flatten(order=flat_order) # s_(i-1,j)
#         s3 = S[1:,:-1].flatten(order=flat_order) # s_(i,j-1)
#         s4 = S[:-1,:-1].flatten(order=flat_order) # s_(i-1,j-1)
#         omega2 = (1/s1 + 1/s2 + 1/s3 + 1/s4)/4
#         return (omega0*omega2) # 1/(alpha*omega2)
#     w = cuqi.distribution.Gamma(shape=omega0, rate=lambda s: w_rate(s), geometry=cuqi.geometry.Image2D((N+1,N+1), order = defect_geom.order), name="w")
#     w.init_point = s_init*np.ones((N+1)**2)


#%%=======================================================================
# Data
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
phantom_geometry = Free(nolayers=nolayers, imagesize=domain, pixeldim = Nphantom, c_coords='cartesian')

# Setup phantom from annulus params
cuqi_phantom_params = cuqi.array.CUQIarray(z_true, is_par=True, geometry = phantom_geometry)

phantom = phantom_geometry.par2fun(cuqi_phantom_params)

# Add cracks not tested in current version!!!
# if add_defects == True:
#     phantom, defectmask = add_cracks(phantom, Nphantom, domain, annulus_params_list)

cuqi_phantom_im = cuqi.array.CUQIarray(phantom, is_par = False, geometry = cuqi.geometry.Image2D((Nphantom, Nphantom)))

#%% Evaluate the exact data
b_true = model_phantom(cuqi_phantom_im)

# Add noise
e0 = np.random.normal(0, 1, np.shape(b_true))
noise_std = rnl*np.linalg.norm(b_true)/np.linalg.norm(e0)
np.random.seed(10)
data_dist_phantom  = cuqi.distribution.Gaussian(mean = model_phantom, sqrtcov = noise_std, geometry=model_phantom.range_geometry)
b_data=data_dist_phantom(x=cuqi_phantom_im).sample()

#%%=======================================================================
# Sampling part 1
#=========================================================================

# data distribution
z0 = z.sample(1)
b1  = cuqi.distribution.Gaussian(mean = A_z, sqrtcov = noise_std, geometry=A.range_geometry, name = "b1")
L = cuqi.likelihood.Likelihood(b1, b_data)
P = cuqi.distribution.Posterior(L, z)
sampler1 = cuqi.sampler.CWMH(P, scale = sample_scale, x0 = z0)

samples1 = sampler1.sample_adapt(N = Ns, Nb = 0)

#%%=======================================================================
# Sampling part 2, NOT TESTED
#=========================================================================

# if defect_recon == True:
#     A_joint = cuqi.model.JointLinearModel([A_z, A_d])
#     b2  = cuqi.distribution.Gaussian(mean = A_joint, sqrtprec = 1/noise_std, geometry=A.range_geometry, name = "b2")
#     P2 = cuqi.distribution.JointDistribution(b2, z, d, s, w)
#     # remember to set init point and step size in z
#     sampler2 = cuqi.sampler.Gibbs(P2(b2=data), {'z': CWMH, 'd': LinearRTO, 's': myIGConjugate, 'w': myGammaSampler})
        
#     samples2 = sampler2.sample(N = Ns, Nb = 0)


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

# colorbar lims when plotting
cmin = -0.2
cmax = 1

#%% sinogram
plt.figure()
b_data.plot(aspect = "auto", interpolation = "none")
plt.colorbar()
plt.savefig(resultpath + 'sinogram.png')

#%% ground_truth
plt.figure()
cuqi_phantom_params.plot(origin = "lower", interpolation = "none", vmin = cmin, vmax = cmax)
plt.colorbar()
plt.savefig(resultpath + 'ground_truth.png')
#
#%% plot chains
plt.figure()
samples1.plot_chain(variable_indices=range(pipe_geom.par_shape[0]))
plt.savefig(resultpath + 'chains.png')

#%% burnin
Nb = 1500
samples_burnin= samples1.burnthin(Nb, Nt = 1)

#%% plot autocorrelations
plt.figure()
samples_burnin.plot_autocorrelation(variable_indices=range(pipe_geom.par_shape[0]), max_lag=None, combined=False)
plt.savefig(resultpath + 'acf.png')

#%%
ess = samples_burnin.compute_ess()
print(ess)

#%% burnthin
#samples_burnthin = samples1.burnthin(Nb, Nt = Nt)

#%% plot chains
plt.figure()
samples_burnin.plot_chain(variable_indices = range(pipe_geom.par_shape[0]))
plt.savefig(resultpath + 'chains_burnin.png')

#%% chains/hist
plt.figure()
samples_burnin.plot_trace(variable_indices=range(pipe_geom.par_shape[0])) 
plt.savefig(resultpath + '1Dmarginals.png')

#%% stats
plt.figure()
samples_burnin.plot_mean(origin = "lower", interpolation = "none", vmin = cmin, vmax = cmax) 
plt.colorbar()
plt.savefig(resultpath + 'mean.png')

#%%
plt.figure()
samples_burnin.plot_ci(plot_par = True, exact = z_true) # crashes without plot_par = True
plt.savefig(resultpath + 'ci.png')

print(samples_burnin.ci_width())

#%% matrix plot
plt.figure()
samples_burnin.plot_pair(variable_indices=range(pipe_geom.par_shape[0])) 
plt.savefig(resultpath + '2Dmarginals.png')



# %%
