# Mat-GAN
#Mat-GAN: Learning the First Principle Calculated Properties for Mixed-atoms Crystals.

Dataset:

CsPb(Br/I)3 in 3*2*1 supercell:

CsPb(Br/I)3 in 2*2*2 supercell:

CsPb(Br/I)3 in the slab model:

Si-Fe-Mn alloy:

Ca6Sn4S14-xOx:github.com/j2hu/MATGANICSS

![Mat-GAN](https://github.com/j2hu/Mat-GAN/blob/master/Mat-GAN-code-flow.png)


#initialization
#############################################################################################################
#############################################################################################################
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

#########################################################################################
#########################################################################################
