# SCIMAI-Gym

[![arXiv](https://img.shields.io/badge/arXiv-2204.09603-b31b1b.svg)](https://arxiv.org/abs/2204.09603)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/frenkowski/SCIMAI-Gym/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

The Jupyter Notebook `SCIMAI-Gym_V1.ipynb` provides the official implementation of the code necessary to replicate the results presented in the paper [A Deep Reinforcement Learning Approach to Supply Chain Inventory Management](http://arxiv.org/abs/2204.09603). 

## Authors Information

**TITLE:** *SCIMAI Gym*  
**AUTHORS:** *Francesco Stranieri and Fabio Stella*  
**INSTITUTION:** *University of Milano-Bicocca*  
**EMAIL:** *francesco.stranieri@unimib.it*

## Citation

```
@misc{stranieri2022deep,
      title={A Deep Reinforcement Learning Approach to Supply Chain Inventory Management}, 
      author={Francesco Stranieri and Fabio Stella},
      year={2022},
      eprint={2204.09603},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Abstract

This paper leverages recent developments in reinforcement learning and deep learning to solve the supply chain inventory management (SCIM) problem, a complex sequential decision-making problem consisting of determining the optimal quantity of products to produce and ship to different warehouses over a given time horizon.
A mathematical formulation of the stochastic two-echelon supply chain environment is given, which allows an arbitrary number of warehouses and product types to be managed.
Additionally, an open-source library that interfaces with deep reinforcement learning (DRL) algorithms is developed and made publicly available for solving the SCIM problem. 
Performances achieved by state-of-the-art DRL algorithms are compared through a rich set of numerical experiments on synthetically generated data. 
The experimental plan is designed and performed, including different structures, topologies, demands, capacities, and costs of the supply chain. 
Results show that the PPO algorithm adapts very well to different characteristics of the environment.
The VPG algorithm almost always converges to a local maximum, even if it typically achieves an acceptable performance level.
Finally, A3C is the fastest algorithm, but just like VPG, it never achieves the best performance when compared to PPO. 
In conclusion, numerical experiments show that DRL performs consistently better than standard reorder policies, such as the static (*s*, *Q*)-policy.
Thus, it can be considered a practical and effective option for solving real-world instances of the stochastic two-echelon SCIM problem.

## Requirements

To install (and import) necessary libraries, run the section:

```setup
Environment Setup
```

The code was tested on:

- [Python](https://www.python.org) 3.7
- [Gym](https://github.com/openai/gym) 0.19.0
- [Ray](https://github.com/ray-project/ray) 1.5.2
- [Ax](https://github.com/facebook/Ax) 0.2.1
- [Matplotlib](https://github.com/matplotlib/matplotlib) 3.4.3

## Supply Chain Environment 

To set up the Supply Chain Environment, run the section:

```env
Reinforcement Learning Classes
```

>📋  To change the configuration of the Supply Chain Environment (such as the number of product types, the number of distribution warehouses, costs, or capacities), edit the sub-section:

```env_conf
Supply Chain Environment Class
```

>📋  To change the global parameters (such as the seed for reproducibility, the number of episodes for the simulations, or the dir to save plots), edit (and run) the section:

```params
Global Parameters
```

Then to initialize the Supply Chain Environment effectively, run the section:

```init
Supply Chain Environment Initialization
```

>❗️  The output of this section will have the following format (verify that the values are the same as the ones you defined):

```init
--- SupplyChainEnvironment --- __init__
product_types_num is 1
distr_warehouses_num is 1
T is 25
d_max is [10]
d_var is [2]
sale_prices is [15]
production_costs is [5]
storage_capacities is [[ 5]
 [10]]
storage_costs is [[2]
 [1]]
transportation_costs is [[0.25]]
penalty_costs is [22.5]
```

Finally, to have some fundamental methods (such as the operational simulator or the plotting methods), run the section:

```methods
Methods
```

## Baselines

To assess the DRL algorithms' performance, we established two different baselines. To initialize the Oracle and the (s, Q)-policy, run the sections:

```baselines
Oracle
(s, Q)-Policy Class
(s, Q)-Policy Config [Ax]
```

>📋  To change the (s, Q)-policy parameters (such as the total trials for the optimization or the number of episodes for each trial), edit the sub-section:

```params
Parameters [Ax]
```

Finally, to have some fundamental methods (such as the methods for the Bayesian Optimization (BO) training or the plotting methods), run the section:

```methods
(s, Q)-Policy Methods [Ax]
```

## Train BO Agents

To train the BO agents, run the section:

```drl_train
(s, Q)-Policy Optimize [Ax]
```

## DRL Config

To change the DRL algorithms' parameters (such as the training episodes or the grace period for the ASHA scheduler), edit (and run) the sub-section:

```drl_config
Parameters [Tune]
```

>📋  To change the DRL algorithms' hyperparameters (such as the neural network structure, the learning rate, or the batch size), edit (and run) the sub-sections:

```drl_hyper
Algorithms [Tune]
A3C Config [Tune]
PG Config [Tune]
PPO Config [Tune]
```

Finally, to have some fundamental methods (such as the methods for the DRL agents' training or the plotting methods), run the section:

```drl_methods
Reinforcement Learning Methods [Tune]
```

## Train DRL Agents

To train the DRL agents, run the section:

```drl_train
Reinforcement Learning Train Agents [Tune]
```

>❗️  We upload the checkpoints of the best training instance for each approach and experiment, which can be used as a pre-trained model. For example, the checkpoint related to the Exp 1 of the 1P3W scenario for the A3C algorithm is available on `/Paper_Results_1P3W/1P3W/Exp_1/1P3W_2021-09-22_15-55-24/ray_results/A3C_2021-09-22_19-56-24/A3C_SupplyChain_2a2cf_00024_24_grad_clip=20.0,lr=0.001,fcnet_hiddens=[64, 64],rollout_fragment_length=100,train_batch_size=2000_2021-09-22_22-34-50/checkpoint_000286/checkpoint-286`.

## Results

To output (and save) the performance (in terms of cumulative profit) and the training time (in minutes) of the DRL algorithms, run the section:

```results
Final Results
```

>❗️  We save the plots of the best training instance for each approach and experiment. For example, the plots related to the Exp 1 of the 1P3W scenario are available on `/Paper_Results_1P3W/1P3W/Exp_1/1P3W_2021-09-22_15-55-24/plots`.

The results obtained should be comparable with those in the paper. For example, for the 1P1W scenario, we achieve the following performance:

|       | **A3C** | **PPO** | **VPG** | **BO**  | **Oracle** |
|-------|---------|---------|---------|---------|------------|
| **Exp 1** | 870±67  | 1213±68 | 885±66  | 1226±71 | 1474±45    |
| **Exp 2** | 1066±94 | 1163±66 | 1100±77 | 1224±60 | 1289±68    |
| **Exp 3** | −36±74  | 195±43  | 12±61   | 101±50  | 345±18     |
| **Exp 4** | 1317±60 | 1600±62 | 883±95  | 1633±39 | 2046±37    |
| **Exp 5** | 736±45  | 838±58  | 789±51  | 870±67  | 966±55     |
