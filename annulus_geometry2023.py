import numpy as np
import matplotlib.pyplot as plt
import cuqi

class PipeGeometry(cuqi.geometry.Geometry):

    def __init__(self, nolayers=1, imagesize=1, pixeldim = 1000, c_coords = 'polar', geom_type = "Free"):
        self.nolayers = nolayers
        self.nodisks = nolayers+1
        self.imagesize = imagesize
        self.pixeldim = pixeldim
        self.c_coords = c_coords  # remember a check
        self.geom_type = geom_type
        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)   
    
    @property
    def fun_shape(self):
        return (self.pixeldim,self.pixeldim)
    
    @property
    def par_shape(self):
        if self.geom_type == "Free":
            return (self.nodisks*4,) 
        elif self.geom_type == "Concentric":
            return (self.nodisks*2+2,)
    
    def disk(self, centerpos1, centerpos2, radius, abscoeff):

        if self.c_coords == 'polar':
            centerpos_angle_rad = centerpos2
            cx = np.cos(centerpos_angle_rad)*centerpos1
            cy = np.sin(centerpos_angle_rad)*centerpos1
        elif self.c_coords == 'cartesian':
            cx = centerpos1
            cy = centerpos2
        else:
            print("Please select c_coords to be 'polar' or 'cartesian'.")

        r1 = (self.xx-cx)**2 + (self.yy-cy)**2
        image = (r1 <= radius**2)

        return image, abscoeff

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)
    
    @property
    def variables(self):

        varnames = []

        if self.geom_type == "Free":
            for i in range(self.nodisks):
                varnames.append("x{}".format(i))
            for i in range(self.nodisks):
                varnames.append("y{}".format(i))

        elif self.geom_type == "Concentric":
            varnames.append("x")
            varnames.append("y")
        
        for i in range(self.nodisks):
            varnames.append("r{}".format(i))
        for i in range(self.nodisks):
            varnames.append("mu{}".format(i))

        self._variables = varnames
        return self._variables

class Free(PipeGeometry):

    def __init__(self, nolayers, imagesize, pixeldim = 1000, c_coords = 'polar'):

        super().__init__(nolayers, imagesize, pixeldim, c_coords, "Free")

    def par2fun(self, params):
        centerpos1 = params[:self.nodisks]
        centerpos2 = params[self.nodisks:2*self.nodisks]
        radii = params[2*self.nodisks:3*self.nodisks]
        abscoeffs = params[3*self.nodisks:]
        #idx_sort = radii.argsort()

        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nodisks)[::-1]:#idx_sort[::-1]:
            tmp, val = self.disk(centerpos1[i], centerpos2[i], np.sum(radii[0:i+1]), abscoeffs[i])
            image[tmp!=0] = val

        return image

class Concentric(PipeGeometry):

    def __init__(self, nolayers, imagesize, pixeldim = 1000, c_coords = 'polar'):

        super().__init__(nolayers, imagesize, pixeldim, c_coords, "Concentric")

    def par2fun(self, params):
        centerpos1 = params[0]
        centerpos2 = params[1]
        radii = params[2:2+self.nodisks]
        abscoeffs = params[2+self.nodisks:]
        #idx_sort = radii.argsort()
        
        image = np.zeros((self.pixeldim, self.pixeldim))
        for i in range(self.nodisks)[::-1]:#idx_sort[::-1]:
            tmp, val = self.disk(centerpos1, centerpos2, np.sum(radii[0:i+1]), abscoeffs[i])
            image[tmp!=0] = val
        return image

class Organizer:
    # should have a print thing that shows the structure

    def __init__(self, pipe_geometry):
        self.pipe_geometry = pipe_geometry

        # initialize cuqipy parameter vector
        self.paramlist = self.pipe_geometry.par_shape[0] * [0]

    def define_pipe_param(self, paramtype, diskno, random, prior = None, value = None):

        if paramtype == "center_x":
            paramno = 0
        elif paramtype == "center_y":
            paramno = 1
        elif paramtype == "radius":
            paramno = 2
        elif paramtype == "abscoeff":
            paramno = 3

        if self.pipe_geometry.geom_type == "Free":
            idx = paramno*(self.pipe_geometry.nodisks) + diskno
        elif self.pipe_geometry.geom_type == "Concentric":
            if paramno < 2:
                idx = paramno
            elif paramno == 2:
                idx = 2 + diskno
            elif paramno == 3:
                idx = 2 + diskno + self.pipe_geometry.nodisks

        tmp = {"paramtype": paramtype,
               "diskno": diskno,
               "random": random,
               "prior": prior,
               "value": value}

        self.paramlist[idx] = tmp

    def get_truth(self):
        center_x = np.zeros(self.pipe_geometry.nodisks)
        center_y = np.zeros(self.pipe_geometry.nodisks)
        radius = np.zeros(self.pipe_geometry.nodisks)
        abscoeff = np.zeros(self.pipe_geometry.nodisks)
        for i in range(self.pipe_geometry.par_shape[0]):
            exec("%s[%d] = %f" % (self.paramlist[i]["paramtype"],self.paramlist[i]["diskno"],self.paramlist[i]["value"]))

        if self.pipe_geometry.geom_type == "Concentric":
            center_x = center_x[0]
            center_y = center_y[0]
        
        out = np.hstack((center_x, center_y, radius, abscoeff))
        return out

    def get_prior(self, name = None):
        # Sum of prior logpdfs
        def _prior_logpdf(*args):
            #print(args)
            out = 0
            for i in range(self.pipe_geometry.par_shape[0]):
                if self.paramlist[i]["random"] == True: # Only sum random variable logpdf's
                    out += self.paramlist[i]["prior"].logpdf(args[0][i])
            return out
        
        def _prior_gradient(*args):
            out = 0
            return out
        
        # sample priors
        def _prior_sample(N=1):
            s = np.zeros((self.pipe_geometry.par_shape[0],N))
            for i in range(self.pipe_geometry.par_shape[0]):
                if self.paramlist[i]["random"] == True:
                    s[i,:] = self.paramlist[i]["prior"].sample(N) # Random variable
                else:
                    s[i,:] = self.paramlist[i]["value"]*np.ones(N) # Not random variable
   
            return np.squeeze(s)

        return cuqi.distribution.UserDefinedDistribution(dim=self.pipe_geometry.par_shape[0], logpdf_func=_prior_logpdf, sample_func = _prior_sample, gradient_func = _prior_gradient, geometry = self.pipe_geometry, name = name)