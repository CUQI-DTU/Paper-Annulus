import numpy as np
import matplotlib.pyplot as plt
import sys
# GitLab CUQI
sys.path.append('../cuqipy/')
import cuqi

class Annulus(cuqi.geometry.Geometry):

    def __init__(self, norings=1, imagesize=1, pixeldim = 1000, c_coords = 'polar', annulus_geom_type = "Free_annuli"):
        self.norings = norings
        self.imagesize = imagesize
        self.pixeldim = pixeldim
        self.c_coords = c_coords  # remember a check
        self.annulus_geom_type = annulus_geom_type
        # Image meshgrid
        c = np.linspace(-self.imagesize/2, self.imagesize/2, self.pixeldim, endpoint=True)
        [self.xx, self.yy] = np.meshgrid(c,c)   

    @property
    def shape(self): #Shape of parameter space
        return (self.norings*5,)

    def annulus(self, centerpos1, centerpos2, innerradius, width, abscoeff):

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
        image = (r1 <= (innerradius + width)**2) & (r1 > innerradius**2)

        return image*abscoeff

    def _plot(self, funvals, **kwargs):
        kwargs.setdefault('cmap', kwargs.get('cmap', "gray"))
        return plt.imshow(funvals, **kwargs)

    def setup_prior(self,annulus_params_list, geometry, name = None):
        
        idx = len(annulus_params_list) * [0]
        dim = len(annulus_params_list)

        for i in range(len(annulus_params_list)):
            anno = annulus_params_list[i].annulusno
            if annulus_params_list[i].paramtype == "center_x":
                paramno = 0
            elif annulus_params_list[i].paramtype == "center_y":
                paramno = 1
            elif annulus_params_list[i].paramtype == "inner_r":
                paramno = 2
            elif annulus_params_list[i].paramtype == "width":
                paramno = 3
            elif annulus_params_list[i].paramtype == "abscoeff":
                paramno = 4

            if self.annulus_geom_type == "Free_annuli":
                idx[i] = anno*5 + paramno
            elif self.annulus_geom_type == "ConcentricConnnected_annuli":
                if paramno < 3:
                    idx[i] = paramno
                elif paramno == 3:
                    idx[i] = 3 + anno
                elif paramno == 4:
                    idx[i] = 3 + self.norings + anno

        # Sum of prior logpdfs
        def _prior_logpdf(*args):
            out = 0
            for i in range(len(annulus_params_list)):
                out += annulus_params_list[i].prior.logpdf(args[0][idx[i]])
            return out

        def _prior_gradient(*args):
            out = 0
            return out

        # sample priors
        def _prior_sample(N=1):
            s = np.zeros((dim,N))
            for i in range(len(annulus_params_list)):
                s[idx[i],:] = annulus_params_list[i].prior.sample(N)
   
            return np.squeeze(s)

        return cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=_prior_logpdf, sample_func = _prior_sample, gradient_func = _prior_gradient, geometry = geometry, name = name)

    @property
    def variables(self):

        varnames = []

        if self.annulus_geom_type == "Free_annuli":
            for i in range(self.norings):
                varnames.append("x{}".format(i))
            for i in range(self.norings):
                varnames.append("y{}".format(i))
            for i in range(self.norings):
                varnames.append("r{}".format(i))

        elif self.annulus_geom_type == "ConcentricConnected_annuli":
            varnames.append("x")
            varnames.append("y")
            varnames.append("r")

        for i in range(self.norings):
            varnames.append("w{}".format(i))
        for i in range(self.norings):
            varnames.append("mu{}".format(i))

        self._variables = varnames
        return self._variables
        
    def annulusparams2paramvec(self,annulus_params_list):
        if self.annulus_geom_type == "Free_annuli":
            center_x = np.zeros(self.norings)
            center_y = np.zeros(self.norings)
            inner_r = np.zeros(self.norings)
            width = np.zeros(self.norings)
            abscoeff = np.zeros(self.norings)
            for i in range(self.norings*5):
                exec("%s[%d] = %f" % (annulus_params_list[i].paramtype,annulus_params_list[i].annulusno,annulus_params_list[i].value))

        elif self.annulus_geom_type == "ConcentricConnected_annuli":
            center_x = np.zeros(1)
            center_y = np.zeros(1)
            inner_r = np.zeros(1)
            width = np.zeros(self.norings)
            abscoeff = np.zeros(self.norings)
            for i in range(3+self.norings*2):
                exec("%s[%d] = %f" % (annulus_params_list[i].paramtype,annulus_params_list[i].annulusno,annulus_params_list[i].value))
        
        out = np.hstack((center_x, center_y, inner_r, width, abscoeff))
        return out

    def cuqiparams2annulusparams(self,cuqi_params_vec):
        annulusparams = []
        if self.annulus_geom_type == "Free_annuli":
            for i in range(self.norings):
                tmp = Annulus_Param("center_x", i, prior=None, value = cuqi_params_vec[i])
                annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("center_y", self.norings+i, prior=None, value = cuqi_params_vec[self.norings+i])
                annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("inner_r", 2*self.norings+i, prior=None, value = cuqi_params_vec[2*self.norings+i])
                annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("width", 3*self.norings+i, prior=None, value = cuqi_params_vec[3*self.norings+i])
                annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("abscoeff", 4*self.norings+i, prior=None, value = cuqi_params_vec[4*self.norings+i])
                annulusparams.append(tmp)

        if self.annulus_geom_type == "ConcentricConnected_annuli":
            tmp = Annulus_Param("center_x", 0, prior=None, value = cuqi_params_vec[0])
            annulusparams.append(tmp)
            tmp = Annulus_Param("center_y", 1, prior=None, value = cuqi_params_vec[1])
            annulusparams.append(tmp)
            tmp = Annulus_Param("inner_r", 2, prior=None, value = cuqi_params_vec[2])
            annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("width", 3+i, prior=None, value = cuqi_params_vec[3+i])
                annulusparams.append(tmp)
            for i in range(self.norings):
                tmp = Annulus_Param("abscoeff", 3+self.norings+i, prior=None, value = cuqi_params_vec[3+self.norings+i])
                annulusparams.append(tmp)

        return annulusparams

class Free_annuli(Annulus):

    def __init__(self, norings, imagesize, pixeldim = 1000, c_coords = 'polar'):

        super().__init__(norings, imagesize, pixeldim, c_coords, "Free_annuli")

    @property
    def shape(self): #Shape of parameter space
        return (self.norings*5,)

    def par2fun(self, params):
        centerpos_lens = params[:self.norings]
        centerpos_angles = params[self.norings:2*self.norings]
        innerradii = params[2*self.norings:3*self.norings]
        widths = params[3*self.norings:4*self.norings]
        abscoeffs = params[4*self.norings:]
        if self.norings == 1:
            image = self.annulus(centerpos_lens, centerpos_angles, innerradii, widths, abscoeffs)
        else:
            image = np.zeros((self.pixeldim, self.pixeldim))
            for i in range(self.norings):
                image += self.annulus(centerpos_lens[i], centerpos_angles[i], innerradii[i], widths[i], abscoeffs[i])
        return image

class ConcentricConnected_annuli(Annulus):

    def __init__(self, norings, imagesize, pixeldim = 1000, c_coords = 'polar'):

        super().__init__(norings, imagesize, pixeldim, c_coords, "ConcentricConnected_annuli")

    @property
    def shape(self): #Shape of parameter space
        return (3+self.norings*2,)

    def par2fun(self, params):
        centerpos_lens = params[0]
        centerpos_angles = params[1] 
        innerradii = params[2]
        widths = params[3:3+self.norings]
        abscoeffs = params[3+self.norings:]
        if self.norings == 1:
            image = self.annulus(centerpos_lens, centerpos_angles, innerradii, widths, abscoeffs)
        else:
            image = np.zeros((self.pixeldim, self.pixeldim))
            for i in range(self.norings):
                image += self.annulus(centerpos_lens, centerpos_angles, innerradii + np.sum(widths[0:i]), widths[i], abscoeffs[i])
        return image

class Annulus_Param:
    def __init__(self, paramtype, annulusno, prior = None, value = None):
        self.paramtype = paramtype
        self.annulusno = annulusno
        self.prior = prior
        self.value = value
