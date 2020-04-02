import torch 
def _logNormalLoss(model, x, t, e):
    
    import numpy as np
    
    shape, scale, logits = model.forward(x)
    
    k_ = shape.expand(x.shape[0], -1)
    b_ = scale.expand(x.shape[0], -1)
        
    ll = 0.
    
    G = model.k
    
    for g in range(G):

        mu = k_[:, g]
        sigma = b_[:, g]
    
        f = - sigma - 0.5*np.log(2*np.pi) - torch.div( (torch.log(t) - mu)**2, 2.*torch.exp(2*sigma)   )
        s = torch.div (torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2) )
        s = 0.5  - 0.5*torch.erf(s)
        s = torch.log(s)


        uncens = np.where(e==1)[0]
        cens   = np.where(e==0)[0]
        
        ll += f[uncens].sum() + s[cens].sum()

    return -ll.mean()


# fitting weibull using only t and e
def _weibullLoss(model, x, t, e):
    
    import numpy as np
    
    torch.manual_seed(0)
    
    shape, scale, logits = model.forward(x, adj=False)
    
    G = model.k

    k_ = shape.expand(x.shape[0], -1)
    b_ = scale.expand(x.shape[0], -1)
    
    ll = 0.
    for g in range(G):

        k = k_[:, g]
        b = b_[:, g]
        

        f = k + b + ((torch.exp(k)-1)*(b+torch.log(t))) - (torch.pow(torch.exp(b)*t , torch.exp(k)))

        s = - (torch.pow(torch.exp(b)*t , torch.exp(k)))

        uncens = np.where(e.cpu().data.numpy() == 1)[0]
        cens   = np.where(e.cpu().data.numpy() == 0)[0]
        
        ll += f[uncens].sum() + s[cens].sum()

    return -ll.mean()


def unConditionalLoss(model, x, t, e):
    
    if model.dist == 'Weibull':
        
        return _weibullLoss(model, x, t, e) 
        
    else if model.dist == 'LogNormal':
        
        return _logNormalLoss(model, x, t, e) 


def _conditionalLogNormalLoss(model, x, t, e, ELBO=True, mean=True, lambd=1e-2, alpha=1., ):
    
    # k = log(shape), b = -log(scale)
    
    import numpy as np
    from torch.nn import LogSoftmax, Softmax
    from torch import lgamma
    
    torch.manual_seed(0)
    
    G = model.k
    
    shape, scale, logits = model.forward(x, adj=True)

    lossf = [] # pdf
    losss = [] # survival
    lossm = [] # sum of squared error of mean / median survival time
    
    k_ = shape
    b_ = scale
    
    for g in range(G):
        
        mu = k_[:, g]
        sigma = b_[:, g]
    

        f = - sigma - 0.5*np.log(2*np.pi) - torch.div( (torch.log(t) - mu)**2, 2.*torch.exp(2*sigma)   )
        s = torch.div (torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2) )
        s = 0.5  - 0.5*torch.erf(s)
        s = torch.log(s)
        

        lossf.append(f)
        losss.append(s)
    
    
    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)
    
    
    if ELBO:
        
        lossg = Softmax(dim=1)(logits) 
        losss = lossg*losss
        lossf = lossg*lossf

        losss = losss.sum(dim=1) 
        lossf = lossf.sum(dim=1) 
    
    else:
        
        lossg = LogSoftmax(dim=1)(logits)
        losss = lossg + losss
        lossf = lossg + lossf
        
        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1) 
        
    lossg = Softmax(dim=1)(logits)    


    
    uncens = np.where(e.cpu().data.numpy() == 1)[0]
    cens = np.where(e.cpu().data.numpy() == 0)[0]



    ll = lossf[uncens].sum() + alpha*losss[cens].sum() 
    return -ll/x.shape[0]


def _conditionalWeibullLoss(model, x, t, e, ELBO=True, mean=True, lambd=1e-2, alpha=1.):
    
    # k = log(shape), b = -log(scale)
    
    import numpy as np
    from torch.nn import LogSoftmax, Softmax
    from torch import lgamma
    
    torch.manual_seed(0)

    G = model.k

    shape, scale, logits = model.forward(x, adj=True)
    

   # print shape, scale, logits
    
    lossf = [] # pdf
    losss = [] # survival
    

    k_ = shape
    b_ = scale
    
    
    for g in range(G):
        
        k = k_[:, g]
        b = b_[:, g]

        f = k + b + ((torch.exp(k)-1)*(b+torch.log(t))) - (torch.pow(torch.exp(b)*t, torch.exp(k)))

        s = - (torch.pow(torch.exp(b)*t , torch.exp(k)))
        
        b_exp = torch.exp(-b) # b_exp = scale
        k_exp = torch.exp(-k) # k_exp = 1/shape
        
  
        lossf.append(f)
        losss.append(s)
    
    
    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)
    
    
    if ELBO:
        
        lossg = Softmax(dim=1)(logits) 
        losss = lossg*losss
        lossf = lossg*lossf

        losss = losss.sum(dim=1) 
        lossf = lossf.sum(dim=1) 
    
    else:
        
        lossg = LogSoftmax(dim=1)(logits)
        losss = lossg + losss
        lossf = lossg + lossf
        
        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1) 
        
    lossg = Softmax(dim=1)(logits)    


    
    uncens = np.where(e.cpu().data.numpy() == 1)[0]
    cens   = np.where(e.cpu().data.numpy() == 0)[0]

    reg = 0
    
    ll = lossf[uncens].sum() + alpha*losss[cens].sum() 

    return -ll/x.shape[0]

def conditionalLoss(model, x, t, e, ELBO=True, mean=True, lambd=1e-2, alpha=1.):
    
    if model.dist == 'Weibull':
        
        return _conditionalWeibullLoss(model, x, t, e, G, ELBO, mean, lambd, alpha) 
        
    else if model.dist == 'LogNormal':
        
        return _conditionalLogNormalLoss(model, x, t, e, G, ELBO, mean, lambd, alpha) 

def _weibull_cdf(model, x, t_horizon):

    import numpy as np
    from torch.nn import Softmax, LogSoftmax
    from scipy.special import gamma
    
    squish = LogSoftmax(dim=1)
    
    G = model.k
        
    shape, scale, logits = model.forward(x, adj=True)
    logits = squish(logits)
    
    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(x.shape[0],1)    

    
    cdfs = []
    pdfs = []
    hazards = []
    
    for j in range(len(t_horizon)):
        
        t = t_horz[:, j]
        
        lcdfs = []
        
        lpdfs = []
    
        for g in range(G):

            k = k_[:, g]
            b = b_[:, g]

            s = - (torch.pow(torch.exp(b)*t , torch.exp(k))) # log survival           
            f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))  - (torch.pow(torch.exp(b)*t, torch.exp(k)))

            lpdfs.append(f)
            lcdfs.append(s)
    
        lcdfs = torch.stack(lcdfs, dim=1)
        lpdfs = torch.stack(lpdfs, dim=1)

        lcdfs = lcdfs+logits
        lpdfs = lpdfs+logits

        lcdfs = torch.logsumexp(lcdfs, dim=1)
        lpdfs = torch.logsumexp(lpdfs, dim=1)

        cdfs.append(lcdfs)
        pdfs.append(lpdfs)
        hazards.append(lpdfs-lcdfs)

    return cdfs

def _lognormal_cdf(model, x, t_horizon):
        
    import numpy as np
    from torch.nn import Softmax, LogSoftmax
    from scipy.special import gamma
    
    squish = LogSoftmax(dim=1)
    
    G = model.k
   
    shape, scale, logits = model.forward(x, adj=True)

    logits = squish(logits)
        
    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(x.shape[0],1)    
    
    cdfs = []
    pdfs = []
    hazards = []
    
    for j in range(len(t_horizon)):
        
        t = t_horz[:, j]
        lcdfs = []
        lpdfs = []
    
        for g in range(G):

            mu = k_[:, g]
            sigma = b_[:, g]

            f = - sigma - 0.5*np.log(2*np.pi) - torch.div( (torch.log(t) - mu)**2, 2.*torch.exp(2*sigma)   )
            s = torch.div (torch.log(t) - mu, torch.exp(sigma)*np.sqrt(2) )
            s = 0.5  - 0.5*torch.erf(s)
            s = torch.log(s)

            lpdfs.append(f)
            lcdfs.append(s)

        lcdfs = torch.stack(lcdfs, dim=1)
        lpdfs = torch.stack(lpdfs, dim=1)

        lcdfs = lcdfs+logits
        lpdfs = lpdfs+logits

        lcdfs = torch.logsumexp(lcdfs, dim=1)
        lpdfs = torch.logsumexp(lpdfs, dim=1)

        cdfs.append(lcdfs)
        pdfs.append(lpdfs)
        hazards.append(lpdfs-lcdfs)

    return cdfs

def predict_cdf(model, x, t_horizon):
    
    if model.dist == 'Weibull':
        
        return _weibull_cdf(model, x, t_horizon)
    
    if model.dist == 'LogNormal':
        
        return _lognormal_cdf(model, x, t_horizon)
        
        