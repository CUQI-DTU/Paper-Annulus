import numpy as np
import matplotlib.pyplot as plt
import cuqi
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(order=True)
class PipeParam():
    sort_index: int = field(init=False, repr=False)
    paramno: int = field(init=False, repr=False)
    paramtype: str
    layerno: int
    random: bool = field(default = True)
    prior: Optional[cuqi.distribution.Distribution] = field(default = None)
    truevalue: Optional[float] = field(default = None)

    def __post_init__(self):
        if self.paramtype == "center_x":
            self.paramno = 0
        elif self.paramtype == "center_y":
            self.paramno = 1
        elif self.paramtype == "radius":
            self.paramno = 2
        elif self.paramtype == "width":
            self.paramno = 3
        elif self.paramtype == "abscoeff":
            self.paramno = 4

        self.sort_index = self.layerno*5 + self.paramno

class PipeParamsCollection():
    def __init__(self, pipeparams_list, pipe_geometry):
        self.pipeparams_list = pipeparams_list
        self.pipe_geometry = pipe_geometry

    def get_prior(self, name = None):

        ordered_pipeparams_list = sorted(self.pipeparams_list, key=lambda pp: (pp.paramno, pp.layerno))

        # Sum of prior logpdfs
        def _prior_logpdf(*args):
            #print(args)
            out = 0
            for i in range(self.pipe_geometry.par_shape[0]):
                if ordered_pipeparams_list[i].random == True: # Only sum random variable logpdf's
                    out += ordered_pipeparams_list[i].prior.logpdf(args[0][i])
            return out
        
        # def _prior_gradient(*args):
        #     out = 0
        #     return out
        
        # sample priors
        def _prior_sample(N=1):
            s = np.zeros((self.pipe_geometry.par_shape[0],N))
            for i in range(self.pipe_geometry.par_shape[0]):
                if ordered_pipeparams_list[i].random == True:
                    s[i,:] = ordered_pipeparams_list[i].prior.sample(N) # Random variable
                else:
                    s[i,:] = ordered_pipeparams_list[i].truevalue*np.ones(N) # Not random variable
   
            return np.squeeze(s)

        return cuqi.distribution.UserDefinedDistribution(dim=self.pipe_geometry.par_shape[0], 
                                                            logpdf_func=_prior_logpdf, 
                                                            sample_func = _prior_sample, 
                                                            geometry = self.pipe_geometry, 
                                                            name = name)

    def get_truth(self):

        ordered_pipeparams_list = sorted(self.pipeparams_list, key=lambda pp: (pp.paramno, pp.layerno))

        out = np.zeros(self.pipe_geometry.par_shape[0])

        for i in range(self.pipe_geometry.par_shape[0]):
            out[i] = ordered_pipeparams_list[i].truevalue

        return cuqi.array.CUQIarray(out, geometry = self.pipe_geometry)

class DiskFree(cuqi.geometry.Geometry):

    def __init__(self, nolayers, imagesize = 1, pixeldim = 1000):

        self.nolayers = nolayers
        self.nodisks = nolayers+1
        self.imagesize = imagesize
        self.pixeldim = pixeldim

        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)

        # Variable names
        varnames = []
        for i in range(self.nodisks):
            varnames.append("cx{}".format(i))
        for i in range(self.nodisks):
            varnames.append("cy{}".format(i))
        for i in range(self.nodisks):
            varnames.append("r{}".format(i))
        for i in range(self.nodisks-1):
            varnames.append("phi{}".format(i+1))
        self._variables = varnames
    
    @property
    def fun_shape(self):
        return (self.pixeldim,self.pixeldim)

    @property
    def par_shape(self):
        return (3+self.nolayers*4,) 
    
    @property
    def variables(self):
        return self._variables

    def indicatorfunc(self, cx, cy, radius):

        r1 = (self.xx-cx)**2 + (self.yy-cy)**2
        image = (r1 <= radius**2)

        return image

    def par2fun(self, params):
        centerpos1 = params[:self.nodisks]
        centerpos2 = params[self.nodisks:2*self.nodisks]
        radii = params[2*self.nodisks:3*self.nodisks]#*10
        abscoeffs = np.insert(params[3*self.nodisks:], 0, 0)#*np.array([1,1/10,1/100,1/10,1/10])

        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nodisks)[::-1]:
            tmp = self.indicatorfunc(centerpos1[i], centerpos2[i], radii[i])
            image[tmp!=0] = abscoeffs[i]

        return image

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)


class DiskConcentric(cuqi.geometry.Geometry):

    def __init__(self, nolayers, imagesize, pixeldim = 1000):

        self.nolayers = nolayers
        self.nodisks = nolayers+1
        self.imagesize = imagesize
        self.pixeldim = pixeldim

        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)

        # Variable names
        varnames = []
        varnames.append("cx")
        varnames.append("cy")
        for i in range(self.nodisks):
            varnames.append("r{}".format(i))
        for i in range(self.nodisks-1):
            varnames.append("phi{}".format(i+1))
        self._variables = varnames

    @property
    def fun_shape(self):
        return (self.pixeldim,self.pixeldim)

    @property
    def par_shape(self):
        return (3+self.nolayers*2,) 
    
    @property
    def variables(self):
        return self._variables
    
    def indicatorfunc(self, cx, cy, radius):

        r1 = (self.xx-cx)**2 + (self.yy-cy)**2
        image = (r1 <= radius**2)

        return image

    def par2fun(self, params):
        centerpos1 = params[0]
        centerpos2 = params[1]
        radii = params[2:2+self.nodisks]
        abscoeffs = np.insert(params[2+self.nodisks:], 0, 0)
        
        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nodisks)[::-1]:#idx_sort[::-1]:
            tmp = self.indicatorfunc(centerpos1, centerpos2, radii[i])
            image[tmp!=0] = abscoeffs[i]
        return image

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)
    

class AnnulusFree(cuqi.geometry.Geometry):

    def __init__(self, nolayers, imagesize = 1, pixeldim = 1000):

        self.nolayers = nolayers
        self.imagesize = imagesize
        self.pixeldim = pixeldim

        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)

        # Variable names
        varnames = []
        for i in range(self.nolayers):
            varnames.append("cx{}".format(i+1))
        for i in range(self.nolayers):
            varnames.append("cy{}".format(i+1))
        for i in range(self.nolayers):
            varnames.append("r{}".format(i))
        for i in range(self.nolayers):
            varnames.append("w{}".format(i+1))
        for i in range(self.nolayers):
            varnames.append("phi{}".format(i+1))
        self._variables = varnames
    
    @property
    def fun_shape(self):
        return (self.pixeldim,self.pixeldim)

    @property
    def par_shape(self):
        return (self.nolayers*5,) 
    
    @property
    def variables(self):
        return self._variables

    def indicatorfunc(self, cx, cy, innerradius, width):

        r1 = (self.xx-cx)**2 + (self.yy-cy)**2
        image = (r1 <= (innerradius + width)**2) & (r1 > innerradius**2)

        return image

    def par2fun(self, params):
        centerpos1 = params[:self.nolayers]
        centerpos2 = params[self.nolayers:2*self.nolayers]
        radii = params[2*self.nolayers:3*self.nolayers]
        widths = params[3*self.nolayers:4*self.nolayers]
        abscoeffs = params[4*self.nolayers:]

        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nolayers)[::-1]:
            tmp = self.indicatorfunc(centerpos1[i], centerpos2[i], radii[i], widths[i])
            image[tmp!=0] += abscoeffs[i] # Adds abscoef to image, no overwriting

        return image

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)


class AnnulusConcentricConnected(cuqi.geometry.Geometry):

    def __init__(self, nolayers, imagesize, pixeldim = 1000):

        self.nolayers = nolayers
        self.imagesize = imagesize
        self.pixeldim = pixeldim

        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)

        # Variable names
        varnames = []
        varnames.append("cx")
        varnames.append("cy")
        varnames.append("r0")
        for i in range(self.nolayers):
            varnames.append("w{}".format(i+1))
        for i in range(self.nolayers):
            varnames.append("phi{}".format(i+1))
        self._variables = varnames

    @property
    def fun_shape(self):
        return (self.pixeldim,self.pixeldim)

    @property
    def par_shape(self):
        return (3+self.nolayers*2,) 
    
    @property
    def variables(self):
        return self._variables
    
    def indicatorfunc(self, cx, cy, innerradius, width):

        r1 = (self.xx-cx)**2 + (self.yy-cy)**2
        image = (r1 <= (innerradius + width)**2) & (r1 > innerradius**2)

        return image

    def par2fun(self, params):
        centerpos1 = params[0]
        centerpos2 = params[1]
        radius = params[2]
        widths = params[3:3+self.nolayers]
        abscoeffs = params[3+self.nolayers:]
        
        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nolayers)[::-1]:#idx_sort[::-1]:
            tmp = self.indicatorfunc(centerpos1, centerpos2, radius + np.sum(widths[0:i]), widths[i])
            image[tmp!=0] += abscoeffs[i]
        return image

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)