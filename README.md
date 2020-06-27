### 1. Project Introduction
The project applied reinforcement learning (RL) to achieve multi-variable optimization in wastewater treatment plants (WWTPs). Specifically, Deep Deterministic Policy Gradient (DDPG) is applied in an AAO-based WWTP to optimize dissolved oxygen (DO) and solid retention time (SRT). The reward function is specially designed as LCA-based form to achieve sustainability optimization. Four scenarios: baseline, LCA-oriented, cost-oriented and effluent-oriented are considered.

### 2. Learning
The DDPG learning process mainly follows the original paper and is introduced in this section. The algorithm is coded with Pytorch under Python 3.7 environment.
Different from the original paper, Gaussian noise $\mathcal{N}$ rather than Ornstein-Uhlenbeck process is used for exploration.
Hyperparameters of DDPG are fine tuned and listed in the Table. Before training, $1,000$ sample points are acquired by Monte Carlo sampling, actions are sampled from uniform distribution. The environment is achieved with the RL toolkit, Gym, developed by OpenAI. The value of DO ranges from 0 to 8 $mg/L$, and wastage rate ranges from 0 to 200 $m^3/d$, i.e. SRT $\geq$ 4.5 $days$. 

$$
\renewcommand\arraystretch{0.8} 
\begin{table}[htpb]
\setlength{\belowcaptionskip}{10pt}
\caption{Hyperparameters of DDPG\\}
\centering
\begin{tabular}{l|l}
\hline
\textbf{Parameter}&\textbf{Value}\\
\hline\hline
\(\text{learning rate}\) & $10^{-5}$ \\
\(\text{Activation function}\) & ReLu \\
\(\text{Neuron of actor network}\) & [32, 64, 32] \\
\(\text{Neuron of critic network}\) & [64, 128, 64] \\
\(\gamma\) & 0.8 \\
\(\tau\) & 0.005 \\
\(\text{buffer capacity}\) & 50 \\
\(\text{batch size}\) & 8 \\
\(\text{noise mean}\) & [0, 0] \\
\(\text{noise variance}\) & [0.5, 10] \\
\(\text{update iteration}\) & 10 \\
\(\text{episode}\) & 2,000 \\
\hline
\end{tabular}
\label{Tab:para}
\end{table}
$$

### 3. Result
The result shows that optimization based on LCA has lowest environmental impacts. The cost scenario tends to lower cost but still has high GHG emissions and eutrophication potential. It is worth mentioning that the upgrading and reconstruction of WWTPs should be implemented with the consideration of other environmental impacts.

<img src="./res/pic/reward.jpg" width = "500" alt="LCA reward" align=center/>

<img src="./res/pic/parameter.jpg" width = "500" alt="parameter" align=center/>

<img src="./res/pic/spider.jpg" width = "500" alt="spider" align=center/>


### 4. Paper
[Paper](http://www.google.com) 

