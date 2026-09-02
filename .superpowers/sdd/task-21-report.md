# Task 21 Report: flowfeat 光流残差核心

## 实现

- 新增 `src/shuttlecut/flowfeat.py`。
- `global_shift` 使用 Shi-Tomasi 特征、PyrLK 和 RANSAC 局部仿射估计整帧平移。
- `local_flow_mag` 对 clamp 后的 bbox 使用指定 Farneback 参数计算平均幅值。
- `residual_action` 按 0.8 倍全局平移范数扣除局部运动，返回最大非负残差。
- 新增 `tests/test_flowfeat.py`，覆盖全局平移、局部运动和静态帧场景。

## 验证

- `.venv/bin/python -m pytest -q tests/test_flowfeat.py`: 3 passed
- `.venv/bin/python -m pytest -q`: 85 passed
- LSP：basedpyright 未安装，环境此前已拒绝安装，无法执行诊断。

## 备注

工作树中已有未跟踪的 `outputs/`，未修改。
