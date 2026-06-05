# -*- coding: utf-8 -*-
"""
打包正式提交件 zip —— 只装 `python meow.py` 运行所需的代码闭包 + 项目报告，
严格排除数据 / 模型缓存 / results / __pycache__ / docx（满足老师"不含数据或模型缓存"）。

提交结构（解包后）：
    MEOW/
      meow/   (入口壳层 + 各 wrapper + DL serve 腿)
      src/    (实现闭包：传统链 + DL 地基)
      config/ (frozen 配置 dataclass)
      models/ (DL 卡带 + registry)
      项目报告.md
老师改数据路径后直接 `cd MEOW/meow && python meow.py` 即可跑。
"""
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "submission_build"
STAGE = BUILD / "MEOW"
ZIP = BUILD / "MEOW_submission.zip"

# 要打进提交件的代码目录（运行 `python meow.py` 的完整闭包；experiments/tests/docs 不进）。
CODE_DIRS = ["meow", "src", "config", "models"]
# meow/ 里这些非代码物随手剔除（需求 docx、解析 txt、缓存、license 可留可不留——这里留 README/LICENSE）。
# .DS_Store 是 macOS 目录元数据垃圾文件（无后缀，故必须按名字显式剔除），绝不能进提交件。
DROP_NAMES = {"__pycache__", ".DS_Store", "_MEOW需求文档_parsed.txt", "MEOW金融时序预测2.0.docx"}
DROP_SUFFIX = {".pyc", ".docx", ".txt", ".h5", ".csv", ".json", ".log"}
KEEP_NONPY = {"README", "LICENSE"}   # meow/ 下保留的非 py 说明文件


def _should_copy(p: Path) -> bool:
    if p.name in DROP_NAMES:
        return False
    if p.suffix in DROP_SUFFIX and p.name not in KEEP_NONPY:
        return False
    return True


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    STAGE.mkdir(parents=True)

    n_files = 0
    for d in CODE_DIRS:
        src_dir = REPO / d
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            if "__pycache__" in p.parts:
                continue
            if not _should_copy(p):
                continue
            rel = p.relative_to(REPO)
            dst = STAGE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n_files += 1

    # 项目报告（草稿）一并打入，提交前用户改名/定稿。
    report = REPO / "docs" / "项目报告_草稿.md"
    if report.exists():
        shutil.copy2(report, STAGE / "项目报告.md")
        n_files += 1

    # 运行说明（含 torch/GPU 可选依赖与回落口径）打入提交件根目录，老师一眼可见怎么跑。
    run_doc = REPO / "docs" / "提交运行说明.md"
    if run_doc.exists():
        shutil.copy2(run_doc, STAGE / "运行说明.md")
        n_files += 1

    # 打 zip。
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in STAGE.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(BUILD))

    size_mb = ZIP.stat().st_size / 1e6
    print("[build] staged {} 文件 -> {}".format(n_files, STAGE))
    print("[build] zip: {} ({:.2f} MB)".format(ZIP, size_mb))
    # 体检：确认没把数据/缓存误打进去。
    bad = [str(p.relative_to(STAGE)) for p in STAGE.rglob("*")
           if p.is_file() and p.suffix in {".h5", ".pyc", ".docx"}]
    print("[build] 违禁文件(应为空): {}".format(bad))
    print("[build] 顶层目录:", sorted(d.name for d in STAGE.iterdir()))


if __name__ == "__main__":
    main()
