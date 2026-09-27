"""保留課程裡熟悉的用法：

    python main.py                         互動輸入任務
    python main.py 查詢6月26日從上海到北京的機票
    python main.py --prompt "……"           相容 OpenManus 的參數寫法

完整命令請用：python -m lingxi --help
"""

import sys

from lingxi.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--prompt"]:
        args = args[1:]
    sys.exit(main(["run", *args]))
