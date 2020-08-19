# Optimization of operation parameters towards sustainable WWTP based on deep reinforcement learning

## 1. Project Introduction
This study used a novel technique, Multi-Agent Deep Reinforcement Learning (MADRL), to optimize dissolved oxygen (DO) and dosage in a hypothetical WWTP. The reward function is specially designed as LCA-based form to achieve sustainability optimization. Four scenarios: baseline, LCA-oriented, cost-oriented and effluent-oriented are considered.

## 2. Learning
The MADDPG learning process mainly follows the original paper and is introduced in this section. 
Different from the original paper, Gaussian noise $\mathcal{N}$ rather than Ornstein-Uhlenbeck process is used for exploration.
Hyperparameters of MADDPG are fine tuned, the total sampling quantity is 25,000. Before training, $500$ sample data are acquired by Monte Carlo sampling, actions are sampled from uniform distribution. 
The value of DO ranges from 0 to 5 $mg/L$, and chemical dosage ranges from 0 to 200 $kg/d$. The first training is implemented under LCA scenario, with the SRT as
15 days. For other scenarios, transfer learning is applied to narrow down required data size by freezing part of the network. 

## 3. Result
The result shows that optimization based on LCA has lowest environmental impacts. The comparison of different SRT indicates that a proper SRT can reduce negative impacts greatly. It is worth mentioning that the retrofitting of WWTPs should be implemented with the consideration of other environmental impacts except cost. Moreover, the comparison between DRL and genetic algorithm (GA) indicates that DRL can solve optimization problems effectively and has great extendibility. 

<img src="./res/pic/reward.jpg" width = "400"/>

<img src="./res/pic/parameter.jpg" width = "450"/>

<img src="./res/pic/spider.jpg" width = "500" alt="spider" align=center/>

## 4. Paper
[Paper](http://www.google.com) 

### Cite the paper as follows:

    @article{
     
    }

