# Optimization of operation parameters towards sustainable WWTP based on deep reinforcement learning

## 1. Project Introduction
This study used a novel technique, Multi-Agent Deep Reinforcement Learning (MADRL), to optimize dissolved oxygen (DO) and dosage in a hypothetical WWTP. The reward function is specially designed as LCA-based form to achieve sustainability optimization. Four scenarios: baseline, LCA-oriented, cost-oriented and effluent-oriented are considered.

## 2. Learning
The MADDPG learning process mainly follows the original paper and is introduced in this section. 
Different from the original paper, Gaussian noise $\mathcal{N}$ rather than Ornstein-Uhlenbeck process is used for exploration.
Hyperparameters of MADDPG are fine tuned, before training, $10,000$ sample data are acquired by Monte Carlo sampling, actions are sampled from uniform distribution. 
The value of DO ranges from 0 to 5 $mg/L$, and chemical dosage ranges from 0 to 800 $kg/d$. The first training is implemented under LCA scenario, with the SRT as
15 days. For other scenarios, transfer learning is applied to narrow down required data size by freezing part of the network. 

- State: The observation of agents includes historical information of five timesteps: (i) influent COD, TN, TP and NH$_3$-N (in ASM state form); (ii) inflow rate; (iii) time; (iv) current DO and dosage respectively. (Note that the codes I provided only include state of one timestep for experiments, which actually does not have stable performance).
- Env: GPS-X and Gym are used to form the environment, the practitioners can also use surrogate models to run the code.
- Reward: Rewards are formulated from LCA and LCCA perspectives, see paper for details.


## 3. Result
The result shows thatoptimization based on LCA has lower environmental impacts compared to baseline scenario, as cost, energy consumption and greenhouse gas emissions reduce to 0.890 CNY/m$^3$-ww, 0.530 kWh/m$^3$-ww, 2.491 kg CO$_2$-eq/m$^3$-ww respectively. The cost-oriented control strategy exhibits comparable overall performance to the LCA-driven strategy since it sacrifices environmental benefits but has lower cost as 0.873 CNY/m$^3$-ww. 

## 4. Paper
[Paper](https://arxiv.org/abs/2008.10417) 

### Cite the paper as follows:

    @misc{chen2021optimal,
      title={Optimal control towards sustainable wastewater treatment plants based on multi-agent reinforcement learning}, 
      author={Kehua Chen and Hongcheng Wang and Borja Valverde-Perezc and Siyuan Zhai and Luca Vezzaro and Aijie Wang},
      year={2021},
      eprint={2008.10417},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
    }

