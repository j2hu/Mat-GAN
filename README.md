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

Ca6Sn4S14-xOx:[MATGANICSS](github.com/j2hu/MATGANICSS)

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
```python
E_Sn=-3.980911
E_S=-2.696603
E_Ca=-1.250692
E_O=-0.867783
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
test and define `if t1pxrd.y[i]>0.25and t1pxrd.x[i]>20` accroding to its crystal structure.

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
    L_matrix = np.asarray((np.dot(RS_xrdmat.T, RS_xrdmat)))
    return L_matrix
```

per step of trianing
------
```python
file_path=train_path
#tfset=[]  
for step in range(1,3):       


    sample_path=[]
    for i in range(1,sample_num+1):
        path_ = random_folder(file_path)
        sample_path.append(path_)

    X_DFT=0

    for subpath_ in sample_path:
    
        try:
            total_energy=get_total_energy(path_)
            X_DFT=linear_transform(total_energy)
        except:
            print(path1_)
         
        train_series.pop(-1)
        train_series.append(X_DFT)
    #update queue with X from D  
    input_series_D=np.asarray(train_series,dtype=np.float64)       
    input_series_D=Variable(torch.from_numpy(input_series_D[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    Dout_real=D1(input_series_D)
    pre_dd.append(Dout_real.data.numpy().mean())
    
    G_input=[]
    for path2_ in sample_path:
        path2_=str(path2_)                
        
        try:
            file2pmg=tomgStructure(path2_)
            L_matrix=GANs_Gmat(file2pmg)
            
        except:
            pass
        G_input.append(L_matrix)
       
    G_input=np.asarray(G_input)
    G_input=G_input[np.newaxis,:,:,:] 
    G_input=np.asarray(G_input,dtype=np.float64) 
    G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
    
    Gout=G1(G_input)
    Gout=round(Gout.data.numpy().mean(),6)   # properties by G

    #update queue with X from G
    train_series.append(Gout)
    train_series.pop(0)
        
    input_series_D=np.asarray(train_series,dtype=np.float64)       
    input_series_D=Variable(torch.from_numpy(input_series_D[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    
    D_out_fake=D1(input_series_D)
    pre_gg.append(D_out_fake.data.numpy().mean())
    
    #loss
    D1_loss=-torch.mean(torch.log(D_out_real)+torch.log(1.-D_out_fake))
    dd=D1_loss.data.numpy().mean()
    mat_Dl.append(dd)
    
    G1_loss=torch.mean(torch.log(1.-D_out_fake))
    gg=G1_loss.data.numpy().mean()
    mat_Gl.append(gg)
    
    #------update Mat-GAN with loss 
    if step%2==0:
        opt_D1.zero_grad()
        D1_loss.backward(retain_graph=True)
        opt_D1.step()
        
        opt_G1.zero_grad()
        G1_loss.backward()
        opt_G1.step()
    else:
        opt_D1.zero_grad()
        D1_loss.backward()
        opt_D1.step()
    

    if step%2==0:
        print(step)
        print('error: ',abs(inverse_transform(Gout)-inverse_transform(X_DFT)))
        
        print(dd)
        print(gg)
        print(prob_Tfactor_mat0.data.numpy().mean())
        print(prob_G1_mat1.data.numpy().mean())
```

Save_dict G and D
----
```python
torch.save(G1.state_dict(),person_path+"/GAN_G_step.pkl") 
torch.save(D1.state_dict(),person_path+"/GAN_D_step.pkl")
```

load dict G and D
----
```python

G1.load_stat_dict(torch.load(person_path+'/GAN_G_step.pkl'))
D1.load_stat_dict(torch.load(person_path+'/GAN_D_step.pkl'))

```
statistics the performance of Mat-GAN in the training set and test set.  
------------

As a example of the binding energy of CSS4O

```python
def get_binding_4O(E_t):
    E_binding= (E_t-6*E_Ca-4*E_Sn-10*E_S-4*E_O)/24
    return E_binding
```


```python
E_Gibbs_test=[]
E_Gmodel_test=[]
abserrset=[]
MSEset=[]
err0set=[]
testfile2=[]
for m1,n1,fname in os.walk(test_path):
    for ieach in n1:
        ieach=test_path+ieach
        testfile2.append(ieach)
start=time.time()        
for path_ in testfile2:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)
        E_Gibbs_test.append(GGG)
        
        g_in=[]
        tomgS=tomgStructure(path_)
        gin=GANs_Gmat(tomgS)
        g_in.append(gin)
        g_in=np.asarray(g_in)
        g_in=g_in[np.newaxis,:,:,:]
        g_in=np.asarray(g_in,dtype=np.float64)
        g_in=Variable(torch.from_numpy(g_in),requires_grad=True)
        Gout=G1(g_in)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_test.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abserrset.append(abserr)
        MSEset.append(mse)
        err0=abs(abserr/GGG)
        err0set.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)


# In[31]:


print(np.asarray(abserrset).mean())

print(np.asarray(MSEset).mean())




# In[ ]:


print(abserrset)


# In[26]:


E_Gibbs_t=[]
E_Gmodel_t=[]
abs_t_errset=[]
err_t_0set=[]
tMSEset=[]
testfile=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        testfile.append(ieach)


# In[25]:





# In[35]:


start=time.time()
#        
for path_ in testfile:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)

        E_Gibbs_t.append(GGG)
        g_in=[]
        tomgS=tomgStructure(path_)
        gin=GANs_Gmat(tomgS)
        g_in.append(gin)
        g_in=np.asarray(g_in)
        g_in=g_in[np.newaxis,:,:,:]
        g_in=np.asarray(g_in,dtype=np.float64)
        g_in=Variable(torch.from_numpy(g_in),requires_grad=True)
        Gout=G1(g_in)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_t.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        tmse=(G_data-GGG)**2
        tMSEset.append(tmse)
        abs_t_errset.append(abserr)
        err0=abs(abserr/GGG)
        err_t_0set.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)



print(np.asarray(abs_t_errset).mean())

print(np.asarray(tMSEset).mean())

```

GCN
====================================
initialization GCN
------------
```python
define the GCN 
class GCNNet(nn.Module):
    def __init__(self, input_size=(sample_num,28,28)):
        super(GCNNet, self).__init__()
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


#####################################################################
#the above parts are the module uesd to the process of initializaiton
#
#####################################################################

###############################################################
#the initialzaiont of G


G1= GCNNet()
opt_G1=torch.optim.Adam(G1.parameters(),lr=0.01)
```

Training GCN
----------------

```python
trainfile=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        trainfile.append(ieach)
i=0

loss_set=[]
for path_ in trainfile:
    X_DFT=[]
    try:
        total_energy=get_total_energy(path_)
        E_DFT=linear_transform(total_energy)
            #print(samp_Gibbs)
    except:
        print(path_)
            
    X_DFT.append(E_DFT)
        
    X_DFT=np.asarray(X_DFT,dtype=np.float64)       
    X_DFT=Variable(torch.from_numpy(X_DFT[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    G_input=[]
           
     
    try:
        tomgS=tomgStructure(path_)
            #print(tomgS)
        L_matrix=GANs_Gmat(tomgS)
            #print(L_matrix)
    except:
        pass
    G_input.append(L_matrix)
       
    G_input=np.asarray(G_input)
    G_input=G_input[np.newaxis,:,:,:] 
    G_input=np.asarray(G_input,dtype=np.float64) 
    G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
    
    Gout=G1(G_input)    
    
    G1_loss=torch.abs(torch.mean(Gout-float(E_DFT)))
    

    opt_G1.zero_grad()
    G1_loss.backward()
    opt_G1.step()
    
    i += 1
    loss_set.append(G1_loss)
    print(i,": ",G1_loss)
```



save dict GCN
-----
```python
torch.save(G1.state_dict(),person_path+"/GCN.pkl") 
```

MAE/MSE
----
```python

Eb_Gibbs_test=[]
Eb_Gmodel_test=[]
abserrsetb=[]
MSEsetb=[]
err0setb=[]
testfile=[]

for m1,n1,fname in os.walk(test_path):
    for ieach in n1:
        ieach=test_path+ieach
        testfile.append(ieach)

start=time.time()        
for path_ in testfile:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)       

        Eb_Gibbs_test.append(GGG)
        
        G_input=[]
        tomgS=tomgStructure(path_)
        L_matrix=GANs_Gmat(tomgS)
        G_input.append(L_matrix)
        G_input=np.asarray(G_input)
        G_input=G_input[np.newaxis,:,:,:]
        G_input=np.asarray(G_input,dtype=np.float64)
        G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
        Gout=G1(G_input)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)

        Eb_Gmodel_test.append(G_data)

        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abserrsetb.append(abserr)
        MSEsetb.append(mse)
        err0=abs(abserr/GGG)
        err0setb.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)

print(np.asarray(abserrsetb).mean())

print(np.asarray(MSEsetb).mean())
###############################
X_DFT_testb=[]
E_Gmodel_testb=[]
abs_t_errsetb=[]
err_t_0setb=[]
tMSEsetb=[]
testfileb=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        testfileb.append(ieach)

start=time.time()        
for path_ in testfileb:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)
        X_DFT_testb.append(GGG)
        
        G_input=[]
        tomgS=tomgStructure(path_)
        L_matrix=GANs_Gmat(tomgS)
        G_input.append(L_matrix)
        G_input=np.asarray(G_input)
        G_input=G_input[np.newaxis,:,:,:]
        G_input=np.asarray(G_input,dtype=np.float64)
        G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
        Gout=G1(G_input)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_testb.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abs_t_errsetb.append(abserr)
        tMSEsetb.append(mse)
        err0=abs(abserr/GGG)
        err_t_0setb.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)

print(np.asarray(abs_t_errsetb).mean())

print(np.asarray(tMSEsetb).mean())
```





