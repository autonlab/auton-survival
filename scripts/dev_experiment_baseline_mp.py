import sys
sys.path.insert(0, '../')

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pandas as pd
import numpy as np

import dsm.prognostics.datasets as datasets
import dsm.prognostics.utilities as utilities

from dsm import DeepSurvivalMachines, DeepConvolutionalSurvivalMachines

import torch
from torch import nn 

import pickle as pkl

from sklearn.model_selection import ParameterGrid

import csv

class CNN(nn.Module):
    
    def __init__(self, hidden, output, dropout=0.5):
        super(CNN, self).__init__()

        self.zeropad = nn.ZeroPad2d((0,0,0,9))
        self.conv1 = nn.Conv2d(1, 10, (10,1), 1,0,1)
        self.conv2 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv3 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv4 = nn.Conv2d(10, 10, (10,1),1,0,1)
        self.conv5 = nn.Conv2d(10,1,(3,1),1,(1,0),1)
        self.fc = nn.Linear(hidden, output)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):

#        print("input:", input.shape)
        input = input.unsqueeze(-1)
        input = torch.transpose(input, 1, 3).contiguous()
        #print("transformed:", input.shape)

        out = self.zeropad(input)
        out = self.zeropad(self.act(self.conv1(out)))
        out = self.zeropad(self.act(self.conv2(out)))
        out = self.zeropad(self.act(self.conv3(out)))
        out = self.act(self.conv4(out))
        out = self.act(self.conv5(out))
        out = self.dropout(out.view(out.size(0), -1))
        out = self.act(self.fc(out))
        
        return out


class DNN(nn.Module):
    
    def __init__(self, inputdim, dropout=0.5):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(inputdim,500)
        self.fc2 = nn.Linear(500,400)
        self.fc3 = nn.Linear(400,300)
        self.fc4 = nn.Linear(300,100)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        
        out = self.dropout(self.act(self.fc1(input)))
        out = self.dropout(self.act(self.fc2(out))) 
        out = self.dropout(self.act(self.fc3(out)))
        out = self.act(self.fc4(out))

        return out


def experiment(params):

    import gc

    representation = params.get('representation', 'NN')
    k = params.get('k', 3) 
    temp = params.get('temp', 0.25)
    dropout = params.get('dropout', 0.5)
    dataset = params.get('dataset', 1)
    random_seed = params.get('random_seed', 0)
    censoring = params.get('censoring', 0.0)
    r_early = params.get('r_early', 1000)
    use_censored = params.get('use_censored', True)
    epochs = params.get('epochs', 100)
    learning_rate = params.get('learning_rate', 1e-3)
    batch_size = params.get('batch_size', 128)
    gpu_loc = params.get('gpu_loc', 1)

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_loc)

    try:

        if os.path.exists("models/"+dict_to_name(params)):
            raise Exception("Model is already trained !!!")

        if representation == 'NN':
            flatten = True
            model = DeepSurvivalMachines(k=k, layers=[500], distribution='Normal', temp=temp,
                                        cuda=True, random_seed=random_seed)

        elif representation == 'DNN':
            flatten = True
            base_model = DNN(450, dropout)
            model = DeepSurvivalMachines(k=k, layers=[100], distribution='Normal',
                                        temp=temp, embedding=base_model,
                                        cuda=True, random_seed=random_seed) 

        elif representation == 'CNN':
            flatten = False
            base_model = CNN(450, 100, dropout)
            model = DeepConvolutionalSurvivalMachines(k=k, hidden=100, distribution='Normal',
                                                    temp=temp, embedding=base_model,
                                                    cuda=True, random_seed=random_seed) 

        train_data, test_data = datasets.load_turbofan(cmapss_folder='../../CMAPSSData/', 
                                                    experiment=dataset, 
                                                    windowsize=30,
                                                    flatten=flatten,
                                                    sequential=False,
                                                    test_last=False,
                                                    censoring=censoring,
                                                    return_censored=use_censored)
        x_tr, t_tr, e_tr = train_data
        x_te, t_te, e_te = test_data
        val_data = (x_tr[:2], t_tr[:2], e_tr[:2]) 
        t_tr[t_tr>r_early] = r_early

        model.fit(x_tr, t_tr, e_tr, batch_size=batch_size, learning_rate=learning_rate,
                iters=epochs, val_data=val_data, early_stop=False)

        predictions = model.predict_mean(x_te)
        test_mse = (np.sqrt(utilities.mean_squared_error(t_te, predictions)))

        predictions = model.predict_mean(x_tr)
        train_mse = (np.sqrt(utilities.mean_squared_error(t_tr, predictions)))

        model_file = open("models/"+dict_to_name(params)+'.pkl', 'wb')
        pkl.dump(model, model_file)
        model_file.flush()
        model_file.close()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return params, test_mse, train_mse

    except Exception as e:

        print("!!!!!!!!!!!!!!!!!!!! EXCEPTION OCCURED !!!!!!!!!!!!!!!!!!!!")
        print(e)
        torch.cuda.empty_cache()
        gc.collect()
        return params, np.nan, np.nan
 


def dict_to_name(dictionary):

    name = ""
    for key in dictionary:
        name+= str(key) + '_' + str(dictionary[key]) + '_'
    return name

def run_all_experiments(param_grid, results_file='results.csv', threads=4):

    from multiprocessing import Pool 

    pool = Pool(threads)

    gpus = param_grid['gpu_loc']
    del param_grid['gpu_loc']
    param_grid = ParameterGrid(param_grid) 
    gpus = gpus*len(param_grid)

    param_grid = list(param_grid)

    for i in range(len(param_grid)):
        param_grid[i]['gpu_loc'] = gpus[i]

    results_file = open(results_file, 'w')
    writer = csv.writer(results_file)

    writer.writerow(['filename', 'dataset', 'representation', 'k', 'temp', 'dropout',
                     'censoring', 'random_seed', 'use_censored', 'learning_rate', 'batch_size',
                     'epochs', 'test_mse', 'train_mse'])


    for output in pool.imap(experiment, param_grid, chunksize=1):

        params, test_mse, train_mse = output

        try:

            if np.isnan(test_mse):
                raise Exception("Ignoring NaN and continuing !!!") 

            print(str(params))
            print(dict_to_name(params))

            writer.writerow([dict_to_name(params),
                            params['dataset'],
                            params['representation'],
                            params['k'],
                            params['temp'],
                            params['dropout'],
                            params['censoring'],
                            params['random_seed'],
                            params['use_censored'],
                            params['learning_rate'],
                            params['batch_size'],
                            params['epochs'],
                            test_mse, 
                            train_mse])

            results_file.flush()

        except Exception as e:
            
            print(e)
            continue

    results_file.close()


param_grid = {
    'k': [1, 2, 3],
    'temp': [0., 0.1],
    'representation': ['NN', 'CNN'],
    #'rep': ['DNN'],
    'dropout': [0.5],
    'dataset': [1],
    'censoring': [0.0, 0.05, 0.1, 0.2],
    'random_seed': [0, 1, 2, 3, 4],
    'use_censored': [True, False],
    'batch_size': [ 128],
    'epochs': [100],
    'learning_rate': [1e-4, 1e-3],
    'gpu_loc': [0, 1, 2, 3]
}

run_all_experiments(param_grid, 'results_test_all.csv', threads=12)




