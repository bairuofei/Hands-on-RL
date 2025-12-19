## 用法


### 配置conda环境
```bash
# 使用提供的 environment.yml 创建环境
conda env create -f environment.yml
# 激活环境
conda activate myenv   # myenv 是 environment.yml 中定义的环境名称
```

### 部署pre-trained model
```bash
python sac_eval.py
```

### 训练model
```bash
python sac_train.py
```
结果保存在`./tb_log/sac_xxxxxxxx_xxxxxx/`目录下。修改`sac_eval.py`中的model读取地址，可以观察模型的运行结果。
```bash
python sac_eval.py
```