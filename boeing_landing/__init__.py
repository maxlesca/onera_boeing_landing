"""Modular Boeing 747 landing pipeline (behavioural cloning).

Step 1:  inertial + GPS  ->  Conv1D  ->  CfC  ->  commands.
Reuses Tudor's recurrent core (ConvCfC / CFC, utils/model_builder,
utils/lightning); only the data preparation differs.
"""
