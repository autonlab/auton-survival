import importlib
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dsm import datasets, DeepSurvivalMachines, DeepConvolutionalSurvivalMachines
import numpy as np
from sksurv.metrics import concordance_index_ipcw, brier_score

x, t, e = datasets.load_dataset('MNIST')
print(x.shape, t.shape, e.shape)
# x = np.random.random((9105,1,100,100))

times = np.quantile(t[e==1], [0.25, 0.5, 0.75]).tolist()

cv_folds = 6
folds = list(range(cv_folds))*10000
folds = np.array(folds[:len(x)])

cis = []
brs = []
for fold in range(cv_folds):
    
    print ("On Fold:", fold)
    
    x_train, t_train, e_train = x[folds!=fold], t[folds!=fold], e[folds!=fold]
    x_test,  t_test,  e_test  = x[folds==fold], t[folds==fold], e[folds==fold]
    
    print (x_train.shape)
    
#     model = DeepSurvivalMachines(distribution='Weibull', layers=[100])
    model = DeepConvolutionalSurvivalMachines(distribution='Weibull', hidden=64)
    model.fit(x_train, t_train, e_train, iters=1, learning_rate=1e-3, batch_size=2001)
    
    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                 dtype=[('e', bool), ('t', int)])
    
    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                 dtype=[('e', bool), ('t', int)])
    
    out_risk = model.predict_risk(x_test, times)
    out_survival = model.predict_survival(x_test, times)

    cis_ = []
    for i in range(len(times)):
        cis_.append(concordance_index_ipcw(et_train, et_test, out_risk[:,i], times[i])[0])
    cis.append(cis_)

    brs.append(brier_score(et_train, et_test, out_survival, times )[1])

print ("Concordance Index:", np.mean(cis,axis=0))
print ("Brier Score:", np.mean(brs,axis=0))
