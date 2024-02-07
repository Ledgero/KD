#运行时的默认设置(default runtime)

default_scope = 'mmdet'# 默认的注册器域名
"""
用户可以在训练、验证和测试循环上添加钩子，以便在运行期间插入一些操作。
"""
default_hooks = dict(
    timer=dict(type='IterTimerHook'), #统计迭代耗时
    logger=dict(type='LoggerHook', interval=20),#打印日志
    # 调用 ParamScheduler 的 step 方法
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 设置 max_keep_ckpts 参数实现最多保存 max_keep_ckpts 个权重;保存最优权重:save_best='auto'
    # 设置开始保存权重的 epoch 数 :save_begin=5
    # 将模型检查点保存间隔设置为按 iter 保存,
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10, save_best='auto', save_begin=20),
    # 确保分布式 Sampler 的 shuffle 生效
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    # 是否启用 cudnn benchmark
    cudnn_benchmark=False,
    # 多进程设置：使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。关闭 opencv 的多线程以避免系统超负荷
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

"""
    type： 日志处理器用于处理运行时日志
    window_size：  日志数值的平滑窗口
    by_epoch 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。
"""
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
# 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
load_from = None
# 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
resume = False
