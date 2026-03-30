## 修改了什么，使diffusion policy适应实际franka数据集

1. config
-基本复用012train_diffusion_transformer_real_world_workspace.yaml
-调整task任务的shape meta，删除env_runner配置

2. workspace：调整与env_runner相关部分

