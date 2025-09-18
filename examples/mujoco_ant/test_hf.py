from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

model_path = load_from_hub(   # 这个函数只返回缓存后的模型路径
    repo_id="buffaX/sac-ant-v4", 
    filename="sac_ant.zip"
)

model = SAC.load(model_path)
print(model.actor)
print(model.critic)

