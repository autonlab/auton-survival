import unittest

from dsm import DeepSurvivalMachines
from dsm.dsm_torch import DeepSurvivalMachinesTorch
from dsm import datasets

import numpy as np

class TestDSM(unittest.TestCase):
  
  def test_support_dataset(self):

    x, t, e = datasets.load_dataset('SUPPORT')
    median_of_t = np.median(t)

    self.assertIsInstance(x, np.ndarray)
    self.assertIsInstance(t, np.ndarray)
    self.assertIsInstance(e, np.ndarray)

    self.assertEqual(x.shape, (9105, 44))
    self.assertEqual(t.shape, (9105,))
    self.assertEqual(e.shape, (9105,))

    model = DeepSurvivalMachines()
    self.assertIsInstance(model, DeepSurvivalMachines)
    model.fit(x, t, e, iters=10)
    self.assertIsInstance(model.torch_model,
                          DeepSurvivalMachinesTorch)
    risk_factor = model.predict_risk(x,median_of_t)
    survival_factor = model.predict_survival(x,median_of_t)
    self.assertEqual((risk_factor+survival_factor).all(), 1)

  def test_pbc_dataset(self):

    x, t, e = datasets.load_dataset('PBC')
    median_of_t = np.median(t)
    
    self.assertIsInstance(x, np.ndarray)
    self.assertIsInstance(t, np.ndarray)
    self.assertIsInstance(e, np.ndarray)

    self.assertEqual(x.shape, (1945, 25))
    self.assertEqual(t.shape, (1945,))
    self.assertEqual(e.shape, (1945,))

    model = DeepSurvivalMachines()
    self.assertIsInstance(model, DeepSurvivalMachines)
    model.fit(x, t, e, iters=10)
    self.assertIsInstance(model.torch_model,
                          DeepSurvivalMachinesTorch)
    risk_factor = model.predict_risk(x,median_of_t)
    survival_factor = model.predict_survival(x,median_of_t)
    self.assertEqual((risk_factor+survival_factor).all(), 1)

  def test_framingham_dataset(self):

    x, t, e = datasets.load_dataset('FRAMINGHAM')
    median_of_t = np.median(t)
    
    self.assertIsInstance(x, np.ndarray)
    self.assertIsInstance(t, np.ndarray)
    self.assertIsInstance(e, np.ndarray)

    self.assertEqual(x.shape, (11627, 18))
    self.assertEqual(t.shape, (11627,))
    self.assertEqual(e.shape, (11627,))

    model = DeepSurvivalMachines()
    self.assertIsInstance(model, DeepSurvivalMachines)
    model.fit(x, t, e, iters=10)
    self.assertIsInstance(model.torch_model,
                          DeepSurvivalMachinesTorch)
    risk_factor = model.predict_risk(x,median_of_t)
    survival_factor = model.predict_survival(x,median_of_t)
    self.assertEqual((risk_factor+survival_factor).all(), 1)

