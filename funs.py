import numpy as np
import cuqi
import astra

#%% ASTRA model

class ASTRAModel(cuqi.model.LinearModel):
    def __init__(self, proj_type, proj_geom, vol_geom):


        # Define image (domain) geometry
        domain_geometry = cuqi.geometry.Image2D((vol_geom["GridRowCount"], vol_geom["GridColCount"]), order = "F")

        # Define sinogram (range) geometry
        num_angles = proj_geom["Vectors"].shape[0] if "Vectors" in proj_geom else proj_geom["ProjectionAngles"].shape[0]
        range_geometry = cuqi.geometry.Image2D((num_angles, proj_geom["DetectorCount"]), order = "F")
        
        # Define linear model
        super().__init__(self._forward_func, self._adjoint_func, range_geometry=range_geometry, domain_geometry=domain_geometry)

        # Create ASTRA projector
        self._proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

        # Store other ASTRA related variables privately
        self._proj_geom = proj_geom
        self._vol_geom = vol_geom

    @property
    def proj_geom(self):
        """ ASTRA projection geometry. """
        return self._proj_geom

    @property
    def vol_geom(self):
        """ ASTRA volume geometry. """
        return self._vol_geom

    @property
    def proj_id(self):
        """ ASTRA projector ID. """
        return self._proj_id

    # CT forward projection
    def _forward_func(self, x: np.ndarray) -> np.ndarray:
        id, sinogram =  astra.create_sino(x, self.proj_id)
        astra.data2d.delete(id)
        return sinogram

    # CT back projection
    def _adjoint_func(self, y: np.ndarray) -> np.ndarray:
        id, volume = astra.create_backprojection(y, self.proj_id)
        astra.data2d.delete(id)
        return volume
    
class FanBeam2DModel(ASTRAModel):
    """ 2D CT model with fan beam.
    
    Assumes a centered beam.

    Parameters
    ------------    
    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.
    
    angles : ndarray
        Angles of projections, in radians.

    source_object_dist : scalar
        Distance between source and object.

    object_detector_dist : scalar
        Distance between detector and object.

    det_spacing : int, default 1
        Detector element size/spacing.

    domain : tuple, default im_size
        Size of image domain.

    proj_type : string
        String indication projection type.
        Can be "line_fanflat", "strip_fanflat", "cuda" etc.

    """
    
    def __init__(self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, 2*np.pi, 60),
        source_object_dist=200,
        object_detector_dist=30,
        det_spacing=None,
        domain=None,
        proj_type='line_fanflat'
        ):

        if domain == None:
            domain = im_size

        if det_spacing is None:
            det_spacing = 1

        proj_geom = astra.create_proj_geom("fanflat", det_spacing, det_count, angles, source_object_dist, object_detector_dist)
        vol_geom = astra.create_vol_geom(im_size[0], im_size[1], -domain[0]/2, domain[0]/2, -domain[1]/2, domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)

# Not tested in current version
def add_cracks(phantom, N, domain, annulus_params_list):
    # radial cracks
    no = 2
    c = np.round(np.array([N/2,N/2]))+np.array([annulus_params_list[0].value, annulus_params_list[1].value])/domain*N
    ang = np.array([np.pi/4, 3*np.pi/4])
    dist = (annulus_params_list[2].value + annulus_params_list[3].value + 0.5*annulus_params_list[5].value)/domain*N
    l = annulus_params_list[5].value*0.7/domain*N*np.ones(no)
    w = l/10
    vals = np.array([0.9, 0]) # Steel, hole

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
        tmpmask = create_polygon([N,N], vertices)
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[i]] = vals[i]
    
    return phantom, defectmask


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
