# SAC Ant Agent

A SAC agent trained on the MuJoCo Ant-v4 environment.

## Training Details
- Algorithm: SAC
- Timesteps: 2.4e6
```
- learning_rate=3e-4,
- buffer_size=1_000_000,      # 经验回放缓冲区大小. 这个参数PPO没有
- batch_size=256,             # 默认256
- tau=0.005,                  # 软更新系数
- gamma=0.99,                 # 折扣因子
- train_freq=1,               # 每步都训练，采集多少个环境步的数据后训练一次
- gradient_steps=1,           # 对replayBuffer中读取到的batch，进行多少次梯度下降更新
```

## Usage

```python
!pip install huggingface_sb3
# login to huggingFace
!huggingface-cli login

from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

model_path = load_from_hub(   # This function only returns the path of the cached model
    repo_id="buffaX/sac-ant-v4", 
    filename="sac_ant.zip"
)

model = SAC.load(model_path)
print(model.actor)
print(model.critic)
```
