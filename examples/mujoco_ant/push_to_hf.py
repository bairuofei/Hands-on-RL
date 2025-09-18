from huggingface_hub import HfApi, create_repo, get_full_repo_name  


model_name = "sac-ant-v4" 
hub_model_id = get_full_repo_name(model_name) 

# create_repo(hub_model_id)


api = HfApi()

# api.upload_file(
#     path_or_fileobj="./sac_ant_model/sac_ant.zip",
#     path_in_repo="sac_ant.zip",  # 这里必须给文件名，要不然不知道想要的文件名叫啥
#     repo_id=hub_model_id,
#     commit_message="Add SAC Ant model",
# )

api.upload_file(
    path_or_fileobj="./sac_ant_model/README.md",
    path_in_repo="README.md",  # 这里必须给文件名，要不然不知道想要的文件名叫啥
    repo_id=hub_model_id,
    commit_message="Add README",
)




print("Model uploaded to HuggingFace Hub!")