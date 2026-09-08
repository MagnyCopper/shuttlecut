# Task 11: evaluate 验收报告

## 实现

- 新增 `shuttlecut.eval.evaluate`：以区间交集 / `max(|D|, |G|)` 公式匹配，支持双边界容差、MAE、missed/extra/fragment/boundary 分类。
- 新增 `EvalReport` dataclass 及空的 `eval` 包初始化文件。
- `shuttlecut eval RALLIES_JSON --gt GT_JSON [--record OUT_MD]` 输出人类可读指标；recall 与 precision 均达到 0.90 时返回 0，否则返回 2。
- `--record` 以 Markdown 表格行追加评测结果。

## TDD 证据

1. 先写 `tests/test_evaluate.py`。
2. `.venv/bin/python -m pytest tests/test_evaluate.py -v`：按预期因 `ModuleNotFoundError: No module named 'shuttlecut.eval'` 失败。
3. 实现后同命令：4 passed。

## 验证

- `.venv/bin/python -m pytest -q`：52 passed。
- CLI 实测：使用 B1 真值生成等同检测 JSON，输出 `recall=1.000 precision=1.000 ... PASS`，并成功追加 Markdown 行。
- LSP：已尝试检查 `evaluate.py` 与 `cli.py`；仓库环境未安装 basedpyright，且此前已拒绝自动安装。

## 变更文件

- `src/shuttlecut/eval/__init__.py`
- `src/shuttlecut/eval/evaluate.py`
- `src/shuttlecut/cli.py`
- `tests/test_evaluate.py`

## Task 11(evaluate) 审查修复

- 将 fragment 判定统一为未匹配检测段与已匹配 GT 存在任意正交集；同步更新验收计划代码块。
- 将 fragment 测试改为排他性断言，并补充完全无重叠检测归类为 extra 的场景。
- TDD：新测试先在旧实现上失败（2 failed, 3 passed），修复后定向测试 5 passed。
- 全量验证：`.venv/bin/python -m pytest -q` 为 53 passed。
- LSP：已尝试检查 `evaluate.py`；环境未安装 basedpyright，且此前已拒绝自动安装。
