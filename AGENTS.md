# ShuttleCut 项目规约(对所有 Agent 生效)

1. **临时文件**:任何临时/中间产物一律写入项目根目录 `temp/`(已被 git 忽略),严禁写入工程以外目录。
2. **Python**:虚拟环境必须初始化在本工程目录内(如 `.venv/`),所有 Python 操作必须在本工程虚拟环境中执行。
3. **依赖本地化**:npm 一律安装到工程范围内(不得 `-g`);模型下载、数据集等同理,统一落在本工程目录内。
4. **全局安装需人工确认**:任何需要全局安装的操作(全局包、系统级工具),必须先用提问工具向用户说明并获同意后才能执行。
5. **子任务阻塞执行**:派出子任务时,主 Agent 必须阻塞等待子任务结束,不使用后台并行。
6. **结构化提问**:所有需要用户回答的问题,必须使用提问工具结构化提问,不使用自由文本追问。

## 项目背景

- 品牌:**ShuttleCut** / GitHub 仓库:`shuttlecut`
- 起源:DJI Osmo Pocket 拍摄的羽毛球视频(4K HEVC,存于 `temp/`),人工剪辑耗时耗力
- 现状:**v1 已交付** —— CLI 出片(process)+5 分钟校准协议(calibrate,实测 P/R 0.9-1.0)

## 工程约定(2026-09 重构后)

### 模型命名(严格执行)
```
models/shuttlecut.pt          # 官方生产模型(process 唯一默认)
models/shuttlecut-<stem>.pt   # 视频专属校准模型(calibrate 产物,自动优先)
models/exp/<tag>.pt           # 实验沙盒(train_heavy 产物,永不参与自动查找)
```
新训练一律 `--out models/exp/…`;晋升 = 评审后复制为 `models/shuttlecut.pt`。

### CLI 契约(v1 版)
- `shuttlecut process INPUT` 输出**恰好 2 个视频**(`<stem>-all-rallies.mp4` + `<stem>-highlights.mp4`);副产物仅显式 `--write-metadata`
- 进度走 stderr(阶段 [1/5]…[5/5]+百分比);stdout 只留最终摘要
- 退出码:0 成功 / 1 处理失败 / 2 用法错误
- 操作知识全部内置于 CLI help(`shuttlecut --help`/`calibrate --help`),无独立手册;错误消息自带恢复路径

### 质量基线(勿重复已判负路线)
- 同场馆/训练集内:P/R 0.83-1.00;校准协议:0.9-1.0(u0010=1.000/1.000)
- 零训练跨场馆:0.1-0.5 抽签(**19 路线实验档案已证明为配置不变式**,见 `docs/eval-history.md`)
- 一切对照实验必须 ≥3 种子(训练方差 ±0.15);GT 一律 look_at 视觉构建(协议=calibrate --help 内标注标准)

### 测试与验证
- `pytest -q` 全绿是提交前提(当前 54 项)
- CLI 改动需 e2e 验证:`process` 后检查输出目录恰好 2 文件
- 有关 GPU 的长任务用分离脚本模式(temp/pipe_*.ps1 + 日志轮询,勿阻塞主会话)
