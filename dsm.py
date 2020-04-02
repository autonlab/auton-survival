import torch.nn as nn
import torch 

class DeepSurvivalMachines(nn.Module):

    def __init__(self, inputdim, k, mlptyp=1, HIDDEN=None, init=False, dist='Weibull'):
        
        super(DeepSurvivalMachines, self).__init__()
        
        shape = 1.
        scale = 1.
        
        self.k = k

        self.mlptype = mlptyp
        self.scale = nn.Parameter(-torch.ones(k))
        self.shape = nn.Parameter(-torch.ones(k))
        
        self.HIDDEN = HIDDEN
        
            
        if mlptyp == 1:
            
            self.gate = nn.Sequential(nn.Linear(inputdim, k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(inputdim, k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(inputdim, k, bias=True))
             
        if mlptyp == 2:
            
            self.gate = nn.Sequential(nn.Linear(HIDDEN[0], k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(HIDDEN[0], k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(HIDDEN[0], k, bias=True))
            
            self.embedding = nn.Sequential(nn.Linear(inputdim, HIDDEN[0], bias=False),
                                      nn.ReLU6())

        if mlptyp == 3:
            
            self.gate = nn.Sequential(nn.Linear(HIDDEN[1], k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(HIDDEN[1], k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(HIDDEN[1], k, bias=True))
            
            self.embedding = nn.Sequential(nn.Linear(inputdim, HIDDEN[0], bias=False),
                                        nn.ReLU6(),
                                        nn.Linear(HIDDEN[0], HIDDEN[1], bias=False),
                                        nn.ReLU6())
        
        if mlptyp == 4:
        
            self.gate = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            
            self.embedding = nn.Sequential(nn.Linear(inputdim, HIDDEN[0], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[0], HIDDEN[1], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[1], HIDDEN[2], bias=False),
                                      nn.ReLU6())
            
        if mlptyp == 5:
        
            self.gate = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            
            self.embedding = nn.Sequential(nn.Linear(inputdim, HIDDEN[0], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[0], HIDDEN[1], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[1], HIDDEN[2], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[2], HIDDEN[3], bias=False),
                                      nn.ReLU6())
       
        if mlptyp == 6:
        
            self.gate = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=False))
            self.scaleg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            self.shapeg = nn.Sequential(nn.Linear(HIDDEN[2], k, bias=True))
            
            self.embedding = nn.Sequential(nn.Linear(inputdim, HIDDEN[0], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[0], HIDDEN[1], bias=False),
                                      nn.ReLU6(),
                                      nn.Linear(HIDDEN[1], HIDDEN[2], bias=False),
                                      nn.ReLU6(),
                                    nn.Linear(HIDDEN[2], HIDDEN[3], bias=False),
                                      nn.ReLU6(),
                                    nn.Linear(HIDDEN[3], HIDDEN[4], bias=False),
                                      nn.ReLU6())
            
        if init is not False:
        
            self.shape.data.fill_(init[0])
            self.scale.data.fill_(init[1])


        self.dist = dist
        
        if self.dist == 'Weibull':
            
            self.act = nn.SELU()
         
        elif self.dist == 'LogNormal':
            
            self.act = nn.Tanh()
    
    def forward(self, x, adj=True):
        
        if self.mlptype == 1:
            
            embed = x
            
        else:
            
            embed = self.embedding(x)
        
        if adj:
            
            return self.act(self.shapeg(embed))+self.shape.expand(x.shape[0],-1),  self.act(self.scaleg(embed))+self.scale.expand(x.shape[0],-1), self.gate(embed)/1000
        
        else:
            
            return  self.shape, self.scale, self.gate(embed)/1000



