#!bin/bash

for i in "projects/policy_based/A2C_CartPole-v1.py" "projects/policy_based/DDPG_Pendulum-v0.py" "projects/policy_based/PPO_Acrobot-v1.py" "projects/policy_based/BipedalWalker-v2.py" "projects/policy_based/LunarLander-v2.py" "projects/policy_based/PPO_Pendulum-v0.py" "projects/policy_based/PPO_RoboschoolHalfCheetah-v1.py" "projects/policy_based/TD3_BipedalWalker-v2.py" "projects/policy_based/TD3_Pendulum-v0.py" "projects/value_based/DoubleDQN_CartPole-v1.py" "projects/value_based/DQN_CartPole-v1.py" "projects/value_based/DuelingDQN_CartPole-v1.py" "projects/value_based/PER_DQN_CartPole-v1.py"
do
    python3 main.py --record --aaa ${i}
done