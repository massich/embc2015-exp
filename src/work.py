Cell 1
------
# setup some stuff
%matplotlib gtk
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

Cell 3
------
def listFlatten(x):
    # This is a list flattening work-arround
    # http://stackoverflow.com/questions/406121/flattening-a-shallow-list-in-python
    return sum(x,[])

# Create the data-frame
df = pd.DataFrame(data)

# Fix the index
patientId = listFlatten([[p_ind]*p_size for p_ind, p_size in enumerate(patient_sizes)])
sampleId = listFlatten([range(p_size) for p_size in patient_sizes.tolist()])
df['patientId'] = pd.Series(patientId, index=df.index)
df['sampleId'] = pd.Series(sampleId, index=df.index)
df.set_index(['patientId', 'sampleId'], inplace=True)

# Fix the columns
data_column_name = ['XX', 'YY', 'T2w']+['T{:02d}'.format(x) for x in range(40)]+['ACD']
micolumns = pd.MultiIndex(levels=[['info', 'T2w', 'DCE', 'ACD'], data_column_name],
                          labels=[[0,0,1]+[2]*40+[3], range(44)],
                          names=['lvl0', 'lvl1'])
df.columns = micolumns


Cell 4
------

for patientId in range(1):
    xx = df.loc[(patientId, slice(None)), :].reset_index()
    label = pd.concat([xx.label]*40, 0).reset_index(drop=True)
    data = pd.melt(xx, 
                   id_vars=['sampleId'], value_vars=['DCE'], 
                   var_name=['drop','DCE']).drop('drop',1)
    data['DCE']=data["DCE"].map(lambda x: int(x[1:]))
    puta  = data
    puta['gt'] = label['gt']
    puta['zone'] = label['zone']

    sns.tsplot(data, time="DCE", unit="sampleId", condition="gt", 
               value="value")

Cell 5 (Use less data)
----------------------
import seaborn as sns
import random

sns.set(style="darkgrid")
num_samples = 1000

def sub_sample(x, n):
    return x.ix[random.sample(x.index,n)]

for patientId in range(17):
    xx = df.loc[(patientId, slice(None)), :].reset_index()
    subsample = random.sample(range(1, len(xx)), num_samples)
    xx = sub_sample(xx,num_samples)
    data = pd.melt(xx,
                   id_vars=['sampleId'], value_vars=['DCE'], var_name=['drop','DCE']
                   ).drop('drop',1)
    data['gt'] = pd.concat([xx[:]['gt']]*40).reset_index(drop=True)
    data['DCE']=data["DCE"].map(lambda x: int(x[1:]))
    
    sns.tsplot(data, 
               time="DCE", unit="sampleId", condition="gt", value="value",
               err_style=["ci_band"], ci=np.linspace(100, 10, 4) )

Cell 5
------
nplots=8
f, ax = plt.subplots(nplots, 1, sharex=True)
for ax, patientId in zip(ax, random.sample(range(1, 17), nplots) ):
    print patientId
    xx = df.loc[(patientId, slice(None)), :].reset_index()
    subsample = random.sample(range(1, len(xx)), num_samples)
    xx = sub_sample(xx,num_samples)
    data = pd.melt(xx,
                   id_vars=['sampleId'], value_vars=['DCE'], var_name=['drop','DCE']
                   ).drop('drop',1)
    data['gt'] = pd.concat([xx[:]['gt']]*40).reset_index(drop=True)
    data['DCE']=data["DCE"].map(lambda x: int(x[1:]))
        
    sns.tsplot(data, 
               time="DCE", unit="sampleId", condition="gt", value="value",
               err_style=["ci_band"], ci=np.linspace(100, 10, 4),
               ax=ax)
