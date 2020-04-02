
import numpy as np


def computeCIScores(model,quantiles, G, x_valid, t_valid, e_valid, t_train, e_train, risk=0):
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sksurv.metrics import concordance_index_ipcw

    #quantiles = [ 43.68333435,  86.8666687 , 146.33333588, 283.54268066]
    

    cdf_preds = predict_cdf(model, x_valid,quantiles, G)
    cdf_preds = [cdf.data.numpy() for cdf in cdf_preds]
       
    t_valid = t_valid.cpu().data.numpy()
    e_valid = e_valid.cpu().data.numpy()
    
    t_train = t_train.cpu().data.numpy()
    e_train = e_train.cpu().data.numpy()
    
    uncensored = np.where(e_valid == 1)[0]
    
    et1 =  np.array([(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', int)])
    et2 =  np.array([(e_valid[i], t_valid[i]) for i in range(len(e_valid))],dtype=[('e', bool), ('t', int)])

    cdf_ci_25 =  concordance_index_ipcw( et1, et2, -cdf_preds[0], tau=quantiles[0]   )
    cdf_ci_50 =  concordance_index_ipcw( et1, et2, -cdf_preds[1],tau= quantiles[1]  )
    cdf_ci_75 =  concordance_index_ipcw( et1, et2, -cdf_preds[2],tau= quantiles[2] )
    cdf_ci_m  =  concordance_index_ipcw( et1, et2, -cdf_preds[3],tau= quantiles[3] )

    return None,None, cdf_ci_25[0],  cdf_ci_50[0],  cdf_ci_75[0], cdf_ci_m[0]
    
def increaseCensoring(e, t, p):
    
    np.random.seed(0)
 
    uncens = np.where(e==1)[0]
    
    mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
    
    toswitch = uncens[mask]

    e[toswitch] = 0
    t_ = t[toswitch]
    
    newt = []
    for t__ in t_:
        
        newt.append(np.random.uniform(1,t__))
        
    t[toswitch] = newt
    
    return e, t


def pretrainDSM(model, x_train, t_train, e_train, x_valid, t_valid, e_valid, \
                n_iter=10000, lr=1e-3, thres=1e-4):
    
    from tqdm import tqdm
    from dsm_loss import unconditionalLoss
    
    dist = model.dist
    
    premodel = DeepSurvivalMachines(x_train.shape[1], 1, init=False, dist=model.dist) 
    
    model.double()

    torch.manual_seed(0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    oldcost = -float('inf')

    patience = 0

    costs = []
    
    for i in tqdm(range(n_iter)):

        optimizer.zero_grad()
    
        loss = unconditionalLoss(premodel, x_train, t_train, e_train) # not conditioned on X
 
        loss.backward()
    
        optimizer.step()
        
        valid_loss = unconditionalLoss(premodel, x_valid, t_valid, e_valid)
        
        valid_loss = valid_loss.detach().cpu().numpy()
        
        costs.append(valid_loss)
    
    
        if np.abs(costs[-1] - oldcost) < thres: 
            
            patience += 1
        
            if patience == 3:
                
                break
        
        oldcost = costs[-1]
        
    return model
    

def trainDSM(model, x_train, t_train, e_train, premodel, x_valid, t_valid, e_valid, \
                        n_iter=10000, lr=1e-3, \
                        ELBO=True, mean=True, lambd=1e-2, alpha=1., thres=1e-4, bs=100):
    
    import numpy as np
    from tqdm import tqdm_notebook as tqdm
    
    from copy import deepcopy
    import gc    
    
    G      = model.k
    mlptyp = model.mlptype
    HIDDEN = model.HIDDEN
    
    
    print ("Pretraining the Underlying Distributions...")
    
    premodel = pretrainDSM(model, x_train, t_train, e_train, x_valid, t_valid, e_valid, \
            n_iter=10000, lr=1e-3, thres=1e-4)
    

    model = WeibullMixture(x_train.shape[1], G, mlptyp=mlptyp, HIDDEN=HIDDEN, \
                           init=(float(premodel.shape[0]), float(premodel.scale[0]) ))
    
    model.double()

    torch.manual_seed(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    oldcost = -float('inf')

    patience = 0

    nbatches = int(x_train.shape[0]/bs)+1
    
    dics = []
        
    costs = []
    for i in tqdm(range(n_iter)):
        
        
        for j in range(nbatches):

            optimizer.zero_grad()

            loss = conditionalWeibullLoss(model, x_train[j*bs:(j+1)*bs], t_train[j*bs:(j+1)*bs], e_train[j*bs:(j+1)*bs], \
                                          G, ELBO=ELBO, mean=mean, lambd=lambd, alpha=alpha)
        
            loss.backward()

            optimizer.step()

        
        valid_loss = conditionalWeibullLoss(model, x_valid, t_valid, e_valid, \
                                            G, ELBO=True, mean=mean, lambd=lambd, alpha=alpha)
        valid_loss = valid_loss.detach().cpu().numpy()
        
        out =  predict_valid(model, G, x_valid, t_valid, e_valid, t_train, e_train)

        valid_loss = np.mean(out[2:])
        
        costs.append(valid_loss)
        
        dics.append(deepcopy(model.state_dict()))

        
        if (costs[-1] < oldcost) == True:

            
            print (valid_loss, out)

            if patience == 2:
                
                maxm= np.argmax(costs)
                    
                print ("max:", maxm)
            
                model.load_state_dict(dics[maxm])
                
                del dics
                    
                gc.collect()
            
                return model, i
            
            else:
                
                patience+=1
        
        else:
            
            patience =0
        
        if i%10==0:
    
            print (valid_loss, out)

        oldcost = costs[-1]
    
    return model, i
    

