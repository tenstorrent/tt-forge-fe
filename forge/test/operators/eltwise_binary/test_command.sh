# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# 
# Commands for running element-wise binary tests
# 

# Run single test
# 
# To run using default parameters
# model,     --bin_model     --> model_1, Note: for binary ops we have 11 models, model_[1-11]
# training,  --bin_train     --> True
# recompute, --bin_recompute --> True
# shape,     --bin_shape     --> [1, 16, 32, 64]
# operation, --bin_op        --> Add
pytest -svv test_eltwise_binary_single.py

# Few examples with passed arguments
pytest -svv test_eltwise_binary_single.py --bin_model model_3 --bin_train True --bin_recompute True --bin_shape '[1, 32, 96, 128]' --bin_op 'Add'
pytest -svv test_eltwise_binary_single.py --bin_model model_1 --bin_train False --bin_recompute True --bin_shape '[1, 32, 256, 128]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute False
pytest -svv test_eltwise_binary_single.py --bin_model model_5 --bin_train False
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_shape '[1, 32, 256, 2048]'

# Issue commands 
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Subtract' --bin_shape '[21, 127, 102, 19]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Subtract' --bin_shape '[29, 30, 15, 51]'

pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Heaviside' --bin_shape '[29, 30, 15, 51]'

pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Max' --bin_shape '[29, 30, 15, 51]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Max' --bin_shape '[114, 120, 95]'

pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[29, 30, 15, 51]'
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[76, 6, 80]'
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[108, 13, 73]'

pytest -svv test_eltwise_binary_single.py --bin_model model_1 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[1, 1, 10000, 10000]'
