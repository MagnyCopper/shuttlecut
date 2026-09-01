import argparse

from shuttlecut import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shuttlecut", description="羽毛球回合自动剪辑")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("process", help="切分回合并导出片段")
    sub.add_parser("label", help="真值标注辅助工具")
    sub.add_parser("eval", help="对比检测结果与真值")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 1
    return 0
