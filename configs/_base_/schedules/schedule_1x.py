# training schedule for 1x
# 训练策略(schedule)
"""
训练和测试的配置
    type='EpochBasedTrainLoop',  # 训练循环的类型
    max_epochs=12,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
"""
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
"""
param_scheduler 字段用于配置参数调度器（Parameter Scheduler）
来调整优化器的超参数（例如学习率和动量）。
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
"""
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
"""
optim_wrapper 是配置优化相关设置的字段。
优化器封装（OptimWrapper）不仅提供了优化器的功能，还支持梯度裁剪、混合精度训练等功能。
type:优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
optimizer:  优化器配置。支持 PyTorch 的各种优化器。
    type='SGD',  # 随机梯度下降优化器
    lr=0.02,  # 基础学习率
    momentum=0.9,  # 带动量的随机梯度下降
    weight_decay=0.0001),  # 权重衰减
"""
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
