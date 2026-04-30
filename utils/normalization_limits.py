# -*- coding: utf-8 -*-
"""
File: normalization_limits.py

Purpose:
    Contains recurrent constants that are used throughout the files for consistency purposes. It contains normally variables from the datasets that are difficult to be processed normally due to the size of the dataset and refer to simulator constants and normalisation upper and lower limits

Key Functionality:
    - Saves constants to be reused throughout the calculations
"""

import torch
GLOBAL_MIN = {'dx': -7.08528922e+00, 
              'dy': -7.04967721e+00, 
              'dz': -5.81295321e+00, 
              'vx': -4.70970619e+00, 
              'vy': -4.78749992e+00, 
              'vz': -2.32903691e+00, 
              'phi': -1.38556128e+00, 
              'theta': -1.32969594e+00, 
              'psi': -4.10490676e+00, 
              'p': -1.56213680e+01, 
              'q': -1.09344636e+01, 
              'r': -4.31710720e+00, 
              'Mx_ext': -3.99990950e-02, 
              'My_ext': -3.99979472e-02, 
              'Mz_ext': -9.99980775e-03,  
              'omega_min': 5.00000000e+03,
              'distance_error': 0.0,
              'attitude_error': 0.0}
GLOBAL_MAX = {'dx': 6.94915898e+00, 
              'dy': 7.00480089e+00, 
              'dz': 5.66367014e+00, 
              'vx': 4.73818446e+00, 
              'vy': 4.79300936e+00, 
              'vz': 2.19937277e+00, 
              'phi': 1.32117585e+00, 
              'theta': 1.39802603e+00, 
              'psi': 4.10320250e+00, 
              'p': 1.26680307e+01, 
              'q': 1.00665052e+01, 
              'r': 4.38410885e+00, 
              'Mx_ext': 3.99996276e-02, 
              'My_ext': 3.99993403e-02, 
              'Mz_ext': 9.99977873e-03,  
              'omega_max': 1.00000000e+04,
              'distance_error': 7.410003182390156,
              'attitude_error': 4.169492767070163}
G = torch.tensor(9.81, dtype=torch.float32)
MASS = torch.tensor(0.389, dtype=torch.float32)
IXX = torch.tensor(0.000906, dtype=torch.float32)
IYY = torch.tensor(0.001242, dtype=torch.float32)
IZZ = torch.tensor(0.002054, dtype=torch.float32)
KX = torch.tensor(1.07933887e-05, dtype=torch.float32)
KY = torch.tensor(9.65250793e-06, dtype=torch.float32)
KZ = torch.tensor(2.7862899e-05, dtype=torch.float32)
KOMEGA = torch.tensor(4.36301076e-08, dtype=torch.float32)
KH = torch.tensor(0.06255013, dtype=torch.float32)
KP = torch.tensor(1.4119331e-09, dtype=torch.float32)
KPV = torch.tensor(-0.00797102, dtype=torch.float32)
KQ = torch.tensor(1.21601884e-09, dtype=torch.float32)
KQV = torch.tensor(0.01292637, dtype=torch.float32)
KR1 = torch.tensor(2.57035545e-06, dtype=torch.float32)
KR2 = torch.tensor(4.10923364e-07, dtype=torch.float32)
KRR = torch.tensor(0.00081293, dtype=torch.float32)
OMEGA_MAX = torch.tensor(1.00000000e+04, dtype=torch.float32)
OMEGA_MIN = torch.tensor(5.00000000e+03, dtype=torch.float32)
TAU = torch.tensor(0.06, dtype=torch.float32)