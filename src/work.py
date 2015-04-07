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
    data[['gt','zone']]=label

    sns.tsplot(data, time="DCE", unit="sampleId", condition="zone", 
               value="value")
    sns.tsplot(data[data.gt=='cancer'], time="DCE", unit="sampleId",
               value="value", color='r')

data[data['gt']=='cancer']
df.label['gt'].
print set(])

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

Cell 6
------

i_want_hue = ['#4C3F30', '#CE53D5', '#74D348', '#7ABEBC', '#C44731',
              '#C8AA83', '#6F70E1', '#537739', '#CBC94A', '#5F88BA',
              '#87D695', '#503359', '#CA4A93', '#C18234', '#C8A7BE',
              '#B54E60', '#966BB8']
myPalette = sns.color_palette(i_want_hue, 17)

df = pd.DataFrame({'a' : [p.a for p in fitted_param],
                   'k' : [p.k for p in fitted_param],
                   'b' : [p.b for p in fitted_param],
                   'q' : [p.q for p in fitted_param], 
                   'v' : [p.v for p in fitted_param],
                   'patient' : range(len(fitted_param))})

g = sns.PairGrid(df, hue="patient", size=2.5,
                palette=mypalette,
                )
                 #vars=["factor", "shift", "v", "std"])
g.map_offdiag(plt.scatter, s=[20*(p+1) for p in patients],
              markers=['o','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'])
g.map_diag(plt.hist)
patients = df.patientid.reset_index()['patientid'].unique()
patients
print 'hi'



Cell variable marker size
-------------------------

n_points = 5
s = [(x+1)*100 for x in range(n_points)]
df = pd.DataFrame({'a' : range(n_points),
                   'b' : range(n_points),
                   'label' : range(n_points),
                   'size' : s})

g = sns.PairGrid(df, hue="label", size=2.5)
g.map_offdiag(plt.scatter, s=s)
g.map_diag(plt.hist)

