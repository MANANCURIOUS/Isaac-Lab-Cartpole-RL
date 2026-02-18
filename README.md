# Isaac-Lab-Cartpole-RL
This project implements and compares Proximal Policy Optimization (PPO) using SB3 and SKRL on Isaac Lab environment 
The objective is to train an RL policy to balance an inverted pendulum on a cart under randomized initial conditions later we have also tried these algorithms for double inverted pendulum.

We:
- Trained PPO using SB3
- Trained PPO using SKRL
- Tuned hyperparameters
- Compared convergence behavior
- Recorded trained policy videos
- Analyzed TensorBoard metrics

# Environment-Details
- Environment: Isaac-Cartpole-v0
- State space: 4D
  - Pole angle
  - Pole angular velocity
  - Cart position
  - Cart velocity
- Action space: 1D continuous force
- Episode length: 5 seconds (~300 steps)
- Termination:
   - Cart out of bounds
   - Pole angle > 90°
   - Time limit reached
 
# Setup Instructions
Create Virtual Environment
```cmd
python -m venv env_isaaclab
env_isaaclab\Scripts\activate
```
Install Isaac Sim
```cmd
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
Clone Isaac Lab
```cmd
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```
Install Isaac Lab Extensions
```cmd
.\isaaclab.bat --install

```
# Running the Cartpole Environment

Random Agent Test 
```
.\isaaclab.bat -p scripts\environments\random_agent.py --task Isaac-Cartpole-v0 --num_envs 32

```
# Training with Stable-Baselines3 (SB3 PPO)
Baseline Training
```
.\isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Cartpole-v0 --num_envs=64
```
The hyperparameters for the baseline trainings were 
```
seed: 42

n_timesteps: 1000000

policy: MlpPolicy

n_steps: 16
batch_size: 4096
n_epochs: 20

gamma: 0.99
gae_lambda: 0.95

learning_rate: 3e-4
clip_range: 0.2

ent_coef: 0.01
vf_coef: 1.0
max_grad_norm: 1.0

policy_kwargs:
  activation_fn: nn.ELU
  net_arch: [32, 32]
  squash_output: false

device: cuda:0
```
Tuned Training Configuration
```
.\isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Cartpole-v0 agent.n_steps=512 
agent.batch_size=64
agent.n_epochs=10
agent.n_timesteps=1500000 
agent.ent_coef=0.005
```
Record Training Video
```
.\isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task Isaac-Cartpole-v0 --video
```
Testing Trained Model 
```
.\isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task Isaac-Cartpole-v0 --use_last_checkpoint
```





























