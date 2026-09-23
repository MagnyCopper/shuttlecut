# 贡献指南 / Contributing

## 快速开始

```bash
git clone https://github.com/MagnyCopper/shuttlecut.git && cd shuttlecut
uv venv .venv --python 3.12 && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
uv pip install -e .
pytest -q                                               # 提交前必须全绿
```

## 规约(详见 AGENTS.md)

- 实验模型一律 `models/exp/`;晋升需 ≥3 种子对照并记入 `docs/eval-history.md`
- CLI 行为改动需 e2e 验证:`process` 后输出目录**恰好 2 个文件**
- 操作知识全部内置于 `--help`,不新增独立手册
- `pytest -q` 全绿是 PR 合并前提

## PR 清单

- [ ] 测试全绿(三平台 CI 通过)
- [ ] 若涉模型/评测:结果记入 eval-history 台账
- [ ] 若涉 CLI:help 文本同步更新
