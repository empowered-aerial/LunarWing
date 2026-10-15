\# Phase 04 — Custom PyTorch Actor–Critic



\## Objective



Implement and test a standalone actor–critic reinforcement-learning component in PyTorch to demonstrate practical understanding of policy networks, state-value estimation, differentiable losses, and gradient-based optimization.



\## Implementation



Implemented a shared multilayer perceptron with separate actor and critic heads. The actor produces categorical action probabilities for LunarLander, while the critic estimates the value of the current observation.



The agent collects complete episodes, calculates discounted Monte Carlo returns, estimates advantages using the critic's value predictions, and updates the network using backpropagation and the Adam optimizer. Gradient clipping is applied to limit excessively large gradient updates.



The combined objective includes policy loss, value-estimation loss, and an entropy bonus to encourage exploration.



\## Experimental Configuration



Environment: LunarLander-v3



Algorithm: Vanilla episodic actor–critic



Training episodes: 100



Random seed: 42



Optimizer: Adam



Learning rate: Not provided in summary.json



Discount factor: Not provided in summary.json



Results



Initial mean evaluation reward: -291.595



Final mean evaluation reward: -565.018



Mean reward over the final 20 training episodes: -212.478



Mean gradient norm before clipping: 475.147



Interpret the results cautiously. This short experiment demonstrates implementation and optimization mechanics; it does not establish statistically reliable performance or superiority over PPO.



\## Reproducibility Artifacts



\- `actor\_critic\_demo.py`

\- `training\_log.csv`

\- `training\_curves.png`

\- `summary.json`

\- `phase04\_actor\_critic.pt`



\## Limitations



The implementation uses Monte Carlo returns, which can have high variance. It does not implement PPO's clipped surrogate objective, generalized advantage estimation, minibatch optimization, or PPO-specific rollout handling. It is an educational actor–critic implementation and is separate from the main LunarWing experimental pipeline.



\## Conclusion



This phase demonstrates hands-on implementation of a neural policy, a state-value function, policy-gradient learning, value regression, entropy regularization, and gradient-based parameter updates in PyTorch. The observed learning outcome is reported separately from successful execution of the implementation.

