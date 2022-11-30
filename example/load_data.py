import os
import numpy as np
from scipy.io import loadmat
from utilities  import plot_model
DATASET_PATH_ = os.getcwd() + "/DataSet"

NAMES_ = [f[:-4] for f in os.listdir(DATASET_PATH_) if f[-4:] == ".mat"]

if os.path.isdir(DATASET_PATH_):
    print ("Using LASA DataSet from {}".format(DATASET_PATH_))
else:
    raise IOError("Could not find LASA Dataset in path: {}".format(DATASET_PATH_))

class _Demo(object):
    """
        Each demo object has attributes:

        pos : 2D cartesian position data. shape: (2,1000)
        t   : corresponding time for each data point. shape: (1,1000)
        vel : 2D cartesian velocity data for motion. shape: (2,1000)
        acc : 2D cartesian acceleration data for motion. shape: (2,1000)

    """
    def __init__(self, demo):


        typelist = str(np.asarray(demo[0][0].dtype))
        typelist = typelist[1:-2].split(', ')
        idx = 0
        for att in typelist:
            if "'O'" in att:
                continue
            else:
                setattr(self, att[2:-1], demo[0][0][idx])
                idx+=1

        assert idx == 5, "Reading data for demo failed"

class _Data(object):
    """
        Data object for each pattern has the following two attributes:

        dt : the average time steps across all demonstrations for this pattern
        demos : array of _Demo objects (len: 7) corresponding the trials for this pattern

    """
    def __init__(self, matdata, name):
        self.name = name
        self.dt = matdata['dt'][0][0]
        self.demos = [_Demo(d) for d in matdata['demos'][0]]

        assert len(self.demos) == 7, "ERROR: Data for matdata could not be read properly."

    def __repr__(self):
        return str({'dt':self.dt, 'demos':self.demos})

    @classmethod
    def get_data(cls, name):
        return cls(loadmat("{}/{}.mat".format(DATASET_PATH_,name)), name)


class _PyLasaDataSet(object):

    def __getattr__(self, name):
        if name in NAMES_:
            return _Data.get_data(name)
        else:
            raise AttributeError("DataSet has no data named '{}'".format(name))
