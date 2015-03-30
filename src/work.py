Cell 1
------
# setup some stuff
%matplotlib inline
%run common.py

# load the data
from scipy.io import loadmat
matfiles = loadmat('../data/raw/all_voxels.mat');
data = np.asmatrix(matfiles['data'])
label = np.ravel(matfiles['label'])
patient_sizes = np.ravel(matfiles['patient_sizes'])

Cell 2
------
import pandas as pd
df = pd.DataFrame(data)
df.columns = ['XX', 'YY', 'T2w']+['T{:02d}'.format(x) for x in range(40)]+['ACD']

Cell 3
------
def listFlatten(x):
    # This is a list flattening work-arround
    # http://stackoverflow.com/questions/406121/flattening-a-shallow-list-in-python
    return sum(x,[])

patientId = listFlatten([[p_ind]*p_size for p_ind, p_size in enumerate(patient_sizes)])
sampleId = listFlatten([range(p_size) for p_size in patient_sizes.tolist()])
df['patientId'] = pd.Series(patientId, index=df.index)
df['sampleId'] = pd.Series(sampleId, index=df.index)
df.set_index(['patientId', 'sampleId'], inplace=True)
df[:3]

micolumns = pd.MultiIndex.from_tuples([('a','foo'),('a','bar'),
                                       ('b','foo'),('b','bah')],
                                      names=['lvl0', 'lvl1'])


