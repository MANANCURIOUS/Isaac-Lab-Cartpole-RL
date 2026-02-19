# Isaac-Lab-Cartpole-RL
This project implements and compares Proximal Policy Optimization (PPO) using SB3 and SKRL on Isaac Lab environment 
The objective is to train an RL policy to balance an inverted pendulum on a cart under randomized initial conditions later we have also tried these algorithms for double inverted pendulum.

## NOTE 
The scripts used were mostly kept the same as received from IsaacSim and Isaac Lab installation. Two of the most prominent RL models are observed. The hyperparameter tuning was done at the cmd such that the viewer can easily run with coherency in the script.

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
# VIDEO RESULTS




https://github.com/user-attachments/assets/22939543-9048-4724-b2f6-4b776c062386

# Tensor Board Graph

<img width="1158" height="570" alt="Screenshot 2026-02-18 193345" src="https://github.com/user-attachments/assets/7137c2f0-21b3-4282-9715-531dc9723fe6" />
<img width="1155" height="580" alt="Screenshot 2026-02-18 193400" src="https://github.com/user-attachments/assets/e8080d1f-ee1c-4914-bdb2-fe5b2ea90671" />
<img width="1151" height="555" alt="Screenshot 2026-02-18 193415" src="https://github.com/user-attachments/assets/a4ac27cd-7f4a-42ac-9ef7-c2f80905b1d0" />

# Training with SKRL PPO

## Baseline Training
```
.\isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Cartpole-v0
```

### Model Architecture 
Policy Network
```
layers: [32, 32]
activations: elu
initial_log_std: 0.0
clip_log_std: True
min_log_std: -20
max_log_std: 2

```
Value Network
```
layers: [32, 32]
activations: elu
```
So baseline architecture was:

- 2 hidden layers
- 32 neurons each
- ELU activation

### PPO Core Hyperparameters
```
rollouts: 32
learning_epochs: 8
mini_batches: 8
discount_factor: 0.99
lambda: 0.95
learning_rate: 5e-4
ratio_clip: 0.2
value_clip: 0.2
entropy_loss_scale: 0.0
value_loss_scale: 2.0
grad_norm_clip: 1.0

```
Training Duration was set for timesteps 4800

## Tuned SKRL Training 
```
.\isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Cartpole-v0 ^
agent.agent.rollouts=64 ^
agent.agent.learning_epochs=10 ^
agent.agent.learning_rate=3e-4 ^
agent.agent.entropy_loss_scale=0.01 ^
agent.trainer.timesteps=300000

```

## Video Of Best Model



https://github.com/user-attachments/assets/9f2cae86-d48b-42b4-9c00-12269b5efcb3


## Result Comparision
<img width="1154" height="583" alt="Screenshot 2026-02-19 011557" src="https://github.com/user-attachments/assets/01bd1e72-e9fa-4d22-9ca8-b02c44af02e5" />
<img width="1158" height="574" alt="image" src="https://github.com/user-attachments/assets/6b8e0b1f-e418-49ee-bb4f-ff27a7ae6b5c" />
<img width="1152" height="547" alt="Screenshot 2026-02-19 011633" src="https://github.com/user-attachments/assets/a4180d7e-2e78-4f07-aa2b-a21e8a486899" />

