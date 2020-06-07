Mat-GAN
========
Learning the First Principle Calculated Properties for Mixed-atoms Crystals.
-------------------------------------------------------------------


Dataset:
----------


CsPb(Br/I)3 in 3*2*1 supercell:

CsPb(Br/I)3 in 2*2*2 supercell:

CsPb(Br/I)3 in the slab model:

Si-Fe-Mn alloy:

Ca6Sn4S14-xOx:github.com/j2hu/MATGANICSS

![Mat-GAN](https://github.com/j2hu/Mat-GAN/blob/master/Mat-GAN-code-flow.png)


#initialization
----------------------
```python
global sample_num, rmat_num, series_num
sample_num=1 #output of G
rmat_num=28  #row nums of the matrix for the input of CNN 
series_num=3 #the number of the element in the queue (D)
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.Dlstm=nn.LSTM(
            input_size=series_num,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),
        )

        
    def forward(self,x):
        D_out,(h_n,h_c)=self.Dlstm(x,None)
        out = self.out(D_out[:,-1,:]) #(batch,time step,input)   
        return out

class GNet(nn.Module):
    
    def __init__(self, input_size=(sample_num,28,28)):
        super(GNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(#(3,28,28)
                in_channels=sample_num,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(32,28,28)
            nn.ReLU(),#->(32,28,28)
            nn.MaxPool2d(kernel_size=2),
        )#->(#->(32,14,14))
        self.conv2=nn.Sequential(#->(32,14,14))
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(64,14,14)
            nn.ReLU(),#->(64,14,14)
            nn.MaxPool2d(kernel_size=2),#->(64,7,7)
        )
        self.out=nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,sample_num),            
        )
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x) #batch(64,7,7)
        x=x.view(x.size(0),-1) #(batch, 64*7*7)
        output=torch.unsqueeze(self.out(x),dim=0)
        return output
        

train_series=[]

for i in range(series_num):

    path_s=random_xxpsk(train_path)
    
    ee1=get_total_energy(path_s)
    
    ee1=linear_transform(ee1)
    
    train_series.append(ee1)
    
G1=GNet()

D1=DNet()

opt_D1=torch.optim.Adam(D1.parameters(),lr=0.01)

opt_G1=torch.optim.Adam(G1.parameters(),lr=0.01)
```

test_dir and train_dir
-----------------------
       
```python
# replace '****' with your path
person_path='*******/GAN'

train_path=person_path+'/train/'

test_path=person_path+'/test/'
```

Random_folder
-------
```python
def random_folder(file_path):
    folder=np.random.choice(glob.glob(file_path +"*"))
    #pos_name=folder+'/POSCAR'
    #out_name=folder+'/OUTCAR'
    return folder
```

Method:extract_energy
-----
E_Sn=-3.980911
E_S=-2.696603
E_Ca=-1.250692
E_O=-0.867783

```python
def get_total_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy_slab=round(np.float64(float(energy_string)),5)
    return energy_slab

def get_binding_4O(E_t):
    E_binding= (E_t-6*E_Ca-4*E_Sn-10*E_S-4*E_O)/24
    return E_binding

def linear_transform(energy):
    global extend_num, move_num
    energy_transform=(energy-move_num)*extend_num
    return energy_transform
def inverse_transform(energy_transform):
    global extend_num, move_num
    energy=energy_transform/extend_num+move_num
    return energy


global extend_num, move_num
extend_num=1000
move_num=get_total_energy(train_path+'1000')
#print(move_num)
```

Method:extract_PXRD and get L_matrix
-----
        test and define `if t1pxrd.y[i]>0.25 and t1pxrd.x[i]>20:`accroding to its crystal structure.
```python
patt_xrd = xrd.XRDCalculator('CuKa')

t1path=train_path+'1000/CONTCAR'
t1=mg.Structure.from_file(t1path)
t1pxrd=patt_xrd.get_pattern(t1)
global base_x,base_y
base_x=[]
base_y=[]
for i in range(len(t1pxrd)):
    if t1pxrd.y[i]>0.25 and t1pxrd.x[i]>20:
        base_x.append(t1pxrd.x[i])
        base_y.append(t1pxrd.y[i])

base_x=base_x[:28]
base_y=base_x[:28]

# the step about Pymatgen POSCAR-> the format of pymatgen
def tomgStructure(folder):
    POSfile=folder+'/CONTCAR'      
    R_mgS=mg.Structure.from_file(POSfile)
    return R_mgS

###
##input_data_to_model
###

def get_xrdmat(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)

    i_column = 28
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i] >0.25  and xrd_data4.x[i]>20:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
        
    xrd_x=abs(xrd_x-base_x)
    xrd_y=10*abs(xrd_y-base_y)
    
    xrd_x=np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=(np.arctan(xrd_y))/180*np.pi
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4

##
################################
#def get_atoms_num(folder2):   #
#    xxx=tomgStructure(folder2)#
#    anum=len(xxx.sites)       # 
#    return anum               #
################################
'''
#pattern ---F
#select one patter(AorF) for per training
t1path=train_path+'2001/CONTCAR'
t1=mg.Structure.from_file(t1path)
t1pxrd=patt_xrd.get_pattern(t1)
global base_x,base_y
base_x=[]
base_y=[]
for i in range(len(t1pxrd)):
    if t1pxrd.y[i]>2 and t1pxrd.y[i]< 20:
        base_x.append(t1pxrd.x[i])
        base_y.append(t1pxrd.y[i])

base_x=base_x[:28]
base_y=base_x[:28]




def get_xrdmat(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)

    i_column = 28
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i] >2 and xrd_data4.y[i] <20:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
        
    #xrd_x=abs(xrd_x-base_x)
    xrd_y=abs(xrd_y-base_y)/100
    
    xrd_x=10*np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=np.exp(np.sqrt(xrd_y))
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4
'''





###
##input_data_for_G
###
def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray((np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS
```




