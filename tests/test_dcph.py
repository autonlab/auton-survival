"""This module contains test functions to
test the DeepCoxPH
models on certain standard datasets.
"""
import unittest
from auton_survival.metrics import survival_regression_metric
from auton_survival.models.cph import DeepCoxPH, DeepCoxPHTorch
from auton_survival import datasets, preprocessing
from sksurv import metrics
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator

import numpy as np
import pandas as pd



class TestDCPH(unittest.TestCase):
  """Base Class for all test functions"""  
  def _get_support_dataset(self):
    return datasets.load_dataset(
        "SUPPORT",
        return_features=True
      )
  
  def _preprocess_data(self, features, feat_dict):
    return preprocessing.Preprocessor().fit_transform(
        features, feat_dict['cat'], feat_dict['num']
      )

  def _init_and_validate_dataset_preprocessing(self):
    outcomes, features, feat_dict = self._get_support_dataset()
    
    self.assertIsInstance(outcomes, pd.DataFrame)
    self.assertIsInstance(features, pd.DataFrame)
    self.assertIsInstance(feat_dict, dict)
    
    # Preprocess (Impute and Scale) the features
    features = self._preprocess_data(features, feat_dict)
    
    x = features
    t = outcomes.time.values
    e = outcomes.event.values
    
    self.assertIsInstance(x, pd.DataFrame)
    self.assertIsInstance(t, np.ndarray)
    self.assertIsInstance(e, np.ndarray)

    self.assertEqual(x.shape, (9105, 38))
    self.assertEqual(t.shape, (9105,))
    self.assertEqual(e.shape, (9105,))
        
    (
      features_train,
      features_test,
      outcomes_train,
      outcomes_test,
    ) = train_test_split(features, outcomes, test_size=0.25, random_state=42)
    
    return features_train, features_test, outcomes_train, outcomes_test
  
  def setUp(self):
    self.data = self._init_and_validate_dataset_preprocessing()
    
  def test_dcph_support_e2e(self):
    """E2E for DCPH with the SUPPORT dataset"""
    (
      features_train,
      features_test,
      outcomes_train,
      outcomes_test,
    ) = self.data

    # Train a Deep Cox Proportional Hazards (DCPH) model
    model = DeepCoxPH(layers=[128, 64, 32])

    self.assertIsInstance(model, DeepCoxPH)
    
    model.fit(
      features_train,
      outcomes_train.time.values,
      outcomes_train.event.values,
      iters=30,
      patience=5,
      vsize=0.1,
    )
    
    self.assertIsInstance(model.torch_model, tuple)
    self.assertIsInstance(model.torch_model[0], DeepCoxPHTorch)
    
    self.assertIs(model.torch_model[0], model.torch_module)
    self.assertIs(model.torch_model[0],  model.torch_model.module)
    
    self.assertIs(model.torch_model[1], model.torch_model.breslow)
    self.assertIsInstance(model.torch_model.breslow, BreslowEstimator)

    # Predict risk at specific time horizons.
    times = [365, 365 * 2, 365 * 4]

    survival_probability = model.predict_survival(features_test, t=times)
    risk_score = model.predict_risk(features_test, t=times)
    
    np.testing.assert_equal((risk_score+survival_probability).all(), 1.0)
       
    ctds = survival_regression_metric(
      "ctd",
      outcomes_test,
      survival_probability,
      times,
      outcomes_train=outcomes_train,
    )
    
    self.assertIsInstance(ctds, list)
    
    for ctd in ctds:
      self.assertIsInstance(ctd, float)
    
    boolean_outcomes = list(
      map(lambda i: True if i == 1 else False, outcomes_test.event.values)
    )
    
    cic = metrics.concordance_index_censored(
      boolean_outcomes,
      outcomes_test.time.values,
      model.predict_time_independent_risk(features_test).squeeze(),
    )
    
    self.assertIsInstance(cic, tuple)
    self.assertIsInstance(cic[0], float)
    
  def test_dcph_should_not_fit_breslow_when_breslow_is_false(self):
    """
    Verify BreslowEstimator is not fitted if breslow=false
    """
    (
      features_train,
      features_test,
      outcomes_train,
      outcomes_test,
    ) = self.data

    # Train a Deep Cox Proportional Hazards (DCPH) model
    model = DeepCoxPH(layers=[128, 64, 32])
    
    model.fit(
      features_train,
      outcomes_train.time.values,
      outcomes_train.event.values,
      iters=30,
      patience=5,
      vsize=0.1,
      breslow=False
    )
    
    times = [365, 365 * 2, 365 * 4]
    
    self.assertIsNone(model.torch_model[1])
    self.assertIsNone(model.torch_model.breslow)
    
    with self.assertRaises(Exception) as cm:
      model.predict_survival(features_test, t=times)
      
    with self.assertRaises(Exception) as cm:
      model.predict_risk(features_test, t=times)