---
name: Bug 报告
about: 报告问题帮助我们改进
labels: bug
body:
  - type: textarea
    id: what-happened
    attributes:
      label: 现象描述
      description: 发生了什么?期望是什么?
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: stderr 日志(含 [1/5]..[5/5] 阶段行)
      render: shell
  - type: input
    id: env
    attributes:
      label: 环境
      description: "OS / GPU / `shuttlecut --version`"
      placeholder: "macOS 15 / M4 / 1.1.0"
  - type: checkboxes
    id: checks
    attributes:
      label: 前置确认
      options:
        - label: 已读 `shuttlecut --help` 与对应子命令 help
          required: true
