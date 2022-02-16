import pandas as pd
import numpy as np

def _encode_cols_index(df):

  columns = df.columns

  # Convert Objects to Strings
  
  for col in columns:
    if df[col].dtype == 'O':
      df.loc[:, col] = df[col].values.astype(str)

  # If Index is Object, covert to String
  if df.index.dtype == 'O':
   df.index = df.index.values.astype(str)

  return df

def load_support(location=''):

  data = pd.read_csv(location+"support2.csv")  

  drop_cols = ['death', 'd.time']

  outcomes = data.copy()
  outcomes['event'] =  data['death']
  outcomes['time'] = data['d.time']
  outcomes = outcomes[['event', 'time']]
  
  cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
  num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
               'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
               'glucose', 'bun', 'urine', 'adlp', 'adls']

  return outcomes, data[cat_feats+num_feats]


def load_actg(location=''):

  data = pd.read_csv(location+"ACTG175.csv", index_col='pidnum')

  drop_cols = ['cens', 'days', 'arms']
  
  outcomes = data.copy()
  outcomes['event'] =  data['cens']
  outcomes['time'] = data['days']
  outcomes = outcomes[['event', 'time']]

  features = data.drop(columns=drop_cols, inplace=False)

  columns = list(set(features.columns)-set(['Unnamed: 0', 'arms']))

  return outcomes, features[columns]



def _load_generic_biolincc_dataset(outcome_tbl, time_col, event_col, features, id_col,
                                   visit_col=None, baseline_visit=None, location=''):

  if not isinstance(baseline_visit, (tuple, set, list)):
    baseline_visit = [baseline_visit]

  # List of all features to extract
  all_features = []
  for feature in features:
    all_features+=features[feature]
  all_features = list(set(all_features)) # Only take the unqiue columns

  if '.sas' in outcome_tbl: outcomes = pd.read_sas(location+outcome_tbl, index=id_col)
  elif '.csv' in outcome_tbl: outcomes = pd.read_csv(location+outcome_tbl, index_col=id_col, encoding='latin-1')
  else: raise NotImplementedError()

  outcomes = outcomes[[time_col, event_col]]

  dataset = outcomes.copy()
  dataset.columns = ['time', 'event']

  for feature in features:
    
    if '.sas' in outcome_tbl: table = pd.read_sas(location+feature, index=id_col)
    elif '.csv' in outcome_tbl: table = pd.read_csv(location+feature, index_col=id_col)
    else: raise NotImplementedError()

    if (visit_col is not None) and (visit_col in table.columns):
      mask = np.zeros(len(table[visit_col])).astype('bool')
      for baseline_visit_ in baseline_visit:
        mask = mask | (table[visit_col]==baseline_visit_)
      table = table[mask] 
    table = table[features[feature]]
    print(table.shape)
    dataset = dataset.join(table)

  outcomes = dataset[['time', 'event']]
  features = dataset[all_features]

  outcomes = _encode_cols_index(outcomes)
  features = _encode_cols_index(features)

  return outcomes, features

def load_crash2(endpoint=None, features=None, location=''):

  if features is None:

    print("No Features Specified!! using default demographic features.")

    features = {'CRASH-2_unblindedcodelist.csv': ['t_code'],
                'CRASH-2_data_1.csv': ['iage', 'isex', 'ninjurytime', 
                                       'iinjurytype', 'isbp', 'irr', 
                                       'icc', 'ihr', 'igcseye', 'igcsmotor', 
                                       'igcsverbal', 'igcs', 'trandomised',
                                       'ddeath', 'ddischarge']} 

  outcomes, features =  _load_generic_biolincc_dataset(outcome_tbl='CRASH-2_data_1.csv', 
                                                time_col='trandomised', 
                                                event_col='ddeath',
                                                features=features,
                                                id_col='ientryid',
                                                location=location+'CRASH2/')

  time_rand =  pd.to_datetime(outcomes['time'], format='%d/%m/%Y')
  time_disch = pd.to_datetime(features['ddischarge'], format='%d/%m/%Y')
  time_death = pd.to_datetime(features['ddeath'], format='%d/%m/%Y')

  features.drop(columns=['ddischarge', 'ddeath', 'trandomised'], inplace=True)

  outcomes['event'] = (~np.isnan(time_death)).astype('int')

  time = np.empty_like(time_rand)
  time[~np.isnan(time_disch)] = time_disch[~np.isnan(time_disch)]
  time[~np.isnan(time_death)] = time_death[~np.isnan(time_death)]

  outcomes['time'] = (time - time_rand).dt.days

  features = features.loc[outcomes['time'].values.astype(int)>=0]
  outcomes = outcomes.loc[outcomes['time'].values.astype(int)>=0]

  return outcomes, features

  if endpoint is None:
    print("No Endpoint specified, using all-cause death as the study endpoint.")
    endpoint = 'death'

  # Set the outcome variable
  event = endpoint

  if event[-3:] == 'dth': time = 'deathfu'
  else: time = event + 'fu'


def load_bari2d(endpoint=None, features=None, location=''):

  if features is None:

    print("No Features Specified!! using default demographic features.")

    features = {'bari2d_bl.sas7bdat': ['strata', 'weight', 
                                       'bmi', 'age', 
                                       'sex', 'race'],
               } 

  if endpoint is None:
    print("No Endpoint specified, using all-cause death as the study endpoint.")
    endpoint = 'death'

  # Set the outcome variable
  event = endpoint

  if event[-3:] == 'dth': time = 'deathfu'
  else: time = event + 'fu'

  return _load_generic_biolincc_dataset(outcome_tbl='bari2d_endpts.sas7bdat', 
                                        time_col=time, 
                                        event_col=event,
                                        features=features,
                                        id_col='id',
                                        location=location+'BARI2D/data/')


def load_topcat(endpoint=None, features=None, location=''):  

  # Default Baseline Features to include: 
  if features is None:

    print("No Features Specified!! using default baseline features.")

    features = {'t003.sas7bdat': ['age_entry', 'GENDER', 
                                  'RACE_WHITE', 'RACE_BLACK', 
                                  'RACE_ASIAN', 'RACE_OTHER',],
                't005.sas7bdat': ['DM'],
                't011.sas7bdat': ['country'],
                'outcomes.sas7bdat':['drug'],
               }

  if endpoint is None:
    print("No Endpoint specified, using all-cause death as the study endpoint.")
    endpoint = 'death'

  # Set the outcome variable
  event = endpoint
  if 'death' in endpoint:
    time = 'time_death'
  else:
    time = 'time_' + event 

  return _load_generic_biolincc_dataset(outcome_tbl='outcomes.sas7bdat', 
                                        time_col=time, 
                                        event_col=event,
                                        features=features,
                                        id_col='ID',
                                        location=location+'TOPCAT/datasets/')

def load_allhat(endpoint=None, features=None, location=''):

  # Default Baseline Features to include: 
  if features is None:

    print("No Features Specified!! using default baseline features.")

    categorical_features = ['RZGROUP', 'RACE', 'HISPANIC', 'ETHNIC', 
                            'SEX', 'ESTROGEN', 'BLMEDS', 'MISTROKE', 
                            'HXCABG', 'STDEPR', 'OASCVD', 'DIABETES', 
                            'HDLLT35', 'LVHECG', 'WALL25', 'LCHD', 
                            'CURSMOKE', 'ASPIRIN', 'LLT', 'RACE2', 
                            'BLMEDS2', 'GEOREGN']

    numeric_features = ['AGE', 'BLWGT', 'BLHGT', 'BLBMI', 'BV2SBP',
                        'BV2DBP', 'EDUCAT', 'APOTAS', 'BLGFR']


    features = {'hyp_vsc.sas7bdat': categorical_features + numeric_features}

  if endpoint is None:
    print("No Endpoint specified, using all-cause death as the study endpoint.")
    endpoint = 'DEATH'

  # Set the outcome variable
  event = endpoint
  if 'CANCER' in endpoint: time = 'DYCANC'
  elif 'EP_CHD' in endpoint: time = 'DYCHD'
  else: time = 'DY' + event  

  full_location = location+'ALLHAT/ALLHAT_v2016a/DATA/Summary/' 

  outcomes, features = _load_generic_biolincc_dataset(outcome_tbl='hyp_vsc.sas7bdat', 
                                                      time_col=time, 
                                                      event_col=event,
                                                      features=features,
                                                      id_col='STUDYID',
                                                      location=full_location)

  outcomes['event'] = 1-(outcomes['event']-1)

  if 'ESTROGEN' in features.columns:

    assert 'SEX' in features.columns, "`SEX` needs to be included if using `ESTROGEN`"

    features['ESTROGEN'][ features['SEX'] == 1.0] = 4.0
    features['ESTROGEN'][features['ESTROGEN'].isna()] = 3.0

  return outcomes, features 


def load_proud(endpoint=None, features=None, location=''):

  raise NotImplementedError()

  if features is None:

    print("No Features Specified!! using default baseline features.")

    categorical_features = ['RZGROUP', 'RACE', 'HISPANIC', 'ETHNIC',
                            'SEX', 'ESTROGEN', 'BLMEDS', 'MISTROKE',
                            'HXCABG', 'STDEPR', 'OASCVD', 'DIABETES',
                            'HDLLT35', 'LVHECG', 'WALL25', 'LCHD',
                            'CURSMOKE', 'ASPIRIN', 'LLT', 'RACE2',
                            'BLMEDS2', 'GEOREGN']

    numeric_features = ['AGE', 'BLWGT', 'BLHGT', 'BLBMI', 'BV2SBP',
                        'BV2DBP', 'EDUCAT', 'APOTAS', 'BLGFR']


    features = {'hyp_vsc.sas7bdat': categorical_features + numeric_features}

  if endpoint is None:
    print("No Endpoint specified, using all-cause death as the study endpoint.")
    endpoint = 'DEATH'

  # Set the outcome variable
  event = endpoint

  if 'CANCER' in endpoint:
    time = 'DYCANC'
  else:
    time = 'DY' + event  

  full_location = location+'ALLHAT/ALLHAT_v2016a/DATA/Summary/' 

  outcomes, features = _load_generic_biolincc_dataset(outcome_tbl='hyp_vsc.sas7bdat', 
                                                      time_col=time, 
                                                      event_col=event,
                                                      features=features,
                                                      id_col='STUDYID',
                                                      location=full_location)

  return outcomes, features 

def load_aimhigh():
  raise NotImplementedError()

def load_amis():
  raise NotImplementedError()

def load_bari():
  raise NotImplementedError()

def load_best():
  raise NotImplementedError()

def load_clever():
  raise NotImplementedError()

def load_oat():
  raise NotImplementedError()

def load_peace():
  raise NotImplementedError()

def load_sprint_pop():
  raise NotImplementedError()

def load_stich():
  raise NotImplementedError()


def load_accord(endpoint=None, features=None, location=''):

  # Default Baseline Features to include:
  if features is None:

    print("No Features Specified!! using default baseline features.")

    features = {
  
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/accord_key.sas7bdat': ['female', 'baseline_age', 'arm', 
                                                                                            'cvd_hx_baseline', 'raceclass',
                                                                                            'treatment'],

                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/bloodpressure.sas7bdat': ['sbp', 'dbp', 'hr'],

                'ACCORD/4-Data Sets - CRFs/4a-CRF Data Sets/f01_inclusionexclusionsummary.sas7bdat': ['x1diab', 'x2mi', 
                'x2stroke', 'x2angina','cabg','ptci','cvdhist','orevasc','x2hbac11','x2hbac9','x3malb','x3lvh','x3sten','x4llmeds',
                'x4gender','x4hdlf', 'x4hdlm','x4bpmeds','x4notmed','x4smoke','x4bmi'],

                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/lipids.sas7bdat': ['chol', 'trig', 'vldl', 'ldl', 'hdl'],
        
                'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/otherlabs.sas7bdat': ['fpg', 'alt', 'cpk', 
                                                                                         'potassium', 'screat', 'gfr',
                                                                                         'ualb', 'ucreat', 'uacr'],
                
            }
           
                
    # outcomes  = {'ACCORD_Private/Data Sets - Analysis 201604/CVDOutcomes_201604.sas7bdat':['censor_po','type_po',                                                   
    #                                             'fuyrs_po', 'fuyrs_po7p', 'censor_tm', 'type_tm', 
    #                                             'fuyrs_tm', 'fuyrs_tm7p', 'censor_cm', 'type_nmi', 'fuyrs_nmi7p', 'censor_nst',
    #                                             'type_nst', 'fuyrs_nst', 'fuyrs_nst7p', 'censor_tst', 'fuyrs_tst', 'fuyrs_tst7p'
    #                                             'censor_chf', 'fuyrs_chf', 'censor_ex', 'type_ex', 'fuyrs_ex', 'fuyrs_ex7p', 
    #                                             'censor_maj', 'type_maj', 'fuyrs_maj7p']
    #               }
        
  if endpoint is None:
    print("No Endpoint specified, using primary study endpoint.")
    endpoint = 'po'

  # Set the outcome variable,
  event = 'censor_'+endpoint
  time = 'fuyrs_'+endpoint

  outcome_tbl = 'ACCORD/3-Data Sets - Analysis/3a-Analysis Data Sets/cvdoutcomes.sas7bdat'

  outcomes, features = _load_generic_biolincc_dataset(outcome_tbl=outcome_tbl,
                                                      time_col=time,
                                                      event_col=event,
                                                      features=features,
                                                      id_col='MaskID',
                                                      location=location,
                                                      visit_col='Visit',
                                                      baseline_visit=(b'BLR', b'S01'))
  outcomes['event'] = 1-outcomes['event']
  outcomes['time'] = outcomes['time']

  outcomes = outcomes.loc[outcomes['time']>1.0]
  features = features.loc[outcomes.index]

  outcomes['time'] = outcomes['time']-1

  return outcomes, features
