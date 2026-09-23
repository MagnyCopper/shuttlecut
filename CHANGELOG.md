# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式。

## [1.1.0] - 2026-09-23

### Added
- V-JEPA 2 探针路径(路线 20):跨场馆零校准 P/R 0.83/0.80(LOEO 5 场馆),process 默认启用
- `--pre/--post` 片头片尾余量可调(修复"球还在飞就切下回合":post 2.0→3.5)
- 模型自动下载(manifest+SHA-256 校验,私有仓自动回退 gh CLI)
- `shuttlecut cache`:可重建缓存查看/清理(dry-run 默认)
- Apple Silicon(mps)与 NVENC/NVDEC 硬件加速支持
- CI(ubuntu/macos/windows 三平台)

### Changed
- 模型解析改为均匀两层(探针→官方 R3D),删除按视频 stem 特例(产品均匀性)
- 切片性能:4K 软编/软解两个病理修复,10 分钟 4K 视频冷启动 45→17.5 分钟

## [1.0.0] - 2026-09-22

- v1 交付:process 恰好 2 输出、calibrate 5 分钟校准协议(实测 0.9-1.0)、
  CLI 完全自说明、双轨版本与 Release 分发、56+ 测试。

## [0.2.0] 及更早

- 19 路线实验档案(零训练跨场馆 0.1-0.5 判负结论)、R3D-18 W64 管线、
  GT 标注协议。完整实验史见 `docs/eval-history.md`。
