import unittest

from dsm import DeepSurvivalMachines
from dsm.dsm_torch import DeepSurvivalMachinesTorch
from dsm import datasets

import numpy as np

class TestDSM(unittest.TestCase):

  def test_dsm(self):

    x, t, e = datasets.load_dataset('SUPPORT')

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
