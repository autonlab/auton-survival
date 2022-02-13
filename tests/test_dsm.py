"""This module contains test functions to
test the accuracy of Deep Survival Machines
models on certain standard datasets.
"""
import unittest

from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.dsm.dsm_torch import DeepSurvivalMachinesTorch
from auton_survival.models.dsm import datasets

import numpy as np

class TestDSM(unittest.TestCase):
  """Base Class for all test functions"""
  def test_support_dataset(self):
    """Test function to load and test the SUPPORT dataset.
    """

    x, t, e = datasets.load_dataset('SUPPORT')
    t_median = np.median(t[e==1])

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
    risk_score = model.predict_risk(x, t_median)
    survival_probability = model.predict_survival(x, t_median)
    np.testing.assert_equal((risk_score+survival_probability).all(), 1.0)

    def test_pbc_dataset(self):
      """Test function to load and test the PBC dataset.
      """

      x, t, e = datasets.load_dataset('PBC')
      t_median = np.median(t[e==1])

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
      risk_score = model.predict_risk(x, t_median)
      survival_probability = model.predict_survival(x, t_median)
      np.testing.assert_equal((risk_score+survival_probability).all(), 1.0)

    def test_framingham_dataset(self):
      """Test function to load and test the Framingham dataset.
      """
      x, t, e = datasets.load_dataset('FRAMINGHAM')
      t_median = np.median(t)

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
      risk_score = model.predict_risk(x, t_median)
      survival_probability = model.predict_survival(x, t_median)
      np.testing.assert_equal((risk_score+survival_probability).all(), 1.0)
