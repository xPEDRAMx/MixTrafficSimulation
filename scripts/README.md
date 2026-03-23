# Training scripts

This folder contains local training/evaluation scripts for active environments in this repo.

Current scripts:

- `sb3_highway_dqn.py`: DQN training and rollout recording on `highway-v0`
- `sb3_highway_ppo.py`: PPO training and evaluation on `highway-v0`
- `sb3_highway_dqn_cnn.py`: DQN + CNN with image observations on `highway-v0`
- `sb3_highway_ppo_transformer.py`: PPO with attention-based custom extractor on `highway-v0`
- `utilsss.py`: notebook-oriented helpers to record/show videos
