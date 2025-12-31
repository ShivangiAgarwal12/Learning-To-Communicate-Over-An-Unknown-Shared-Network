# Learning-To-Communicate-Over-An-Unknown-Shared-Network
Learning To Communicate Over An Unknown Shared Network
As robots (edge-devices, agents) find uses in an increasing number of settings and edge-cloud resources become pervasive, wireless networks will often be shared by flows of data traffic that result from communication between agents and their corresponding edge-cloud nodes (cloud compute or data resource accessed by an agent). In such a setting, any agent communicating with the edge-cloud is unaware of the state of the network resource, which evolves in response to not just the agent’s own communication at any given time but also to communication by the other agents, which stays unknown to the agent. 
We address the challenge of an agent learning a policy that allows it to decide whether or not to communicate with its cloud node, using limited feedback it obtains from its own attempts to communicate, with the goal of optimizing its utility. The policy must generalize well to any number of other agents sharing the network and must not be trained for any particular network configuration. Our proposed policy is a deep reinforcement learning model Query Net (QNet) that we train using a proposed simulation-to-real framework. 

Paper Link: [Learning To Communicate Over An Unknown Shared Network](https://dl.acm.org/doi/10.1145/3786203)

###### Code walk-through
1. **main_n_steps_general_mse_range.py** : Main file for training.
2. **env_with_q_backlog_range.py** : Vehicular dynamics in format of gym environment, contains step and reset functions.
3. **queue_model_mod.py** : Network queue simulation model
4. **core_v5.py** : Contains neural network models for LSTM, Actor and Critic.
5. **DSAC_RL_v2.py** : Update functions for SAC, LSTM, LSTM replay buffer, n-step replay buffer
6. **config.py**: file with all hyperparameters
