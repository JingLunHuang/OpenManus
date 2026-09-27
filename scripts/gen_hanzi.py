"""重新產生 lingxi/hanzi.py：繁→簡單字對照（長度不變）＋ 台灣用語→大陸用語詞彙對照。

    pip install opencc-python-reimplemented     # 只有產生時需要，執行期不依賴
    python scripts/gen_hanzi.py

資料來源：OpenCC（Apache-2.0）的 TSCharacters / TWVariantsRev / HKVariantsRev / TWPhrasesRev。
"""

import os
import sys
from pathlib import Path

import opencc

ROOT = Path(__file__).resolve().parent.parent
DICT = Path(opencc.__file__).parent / "dictionary"
# OpenCC 的對應偏書面，網頁 UI 上更常見的是這些說法
OVERRIDES_TW = {"預設": "默認", "帳號": "賬號", "帳戶": "賬戶", "登入": "登錄", "登出": "退出"}


def read(name: str) -> list[tuple[str, str]]:
    pairs = []
    for line in (DICT / name).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            pairs.append((key, value.split(" ")[0]))
    return pairs


def main() -> None:
    chars: dict[str, str] = {}
    for t, s in read("TSCharacters.txt"):
        if len(t) == 1 and len(s) == 1 and t != s:
            chars[t] = s
    for variant, standard in read("TWVariantsRev.txt") + read("HKVariantsRev.txt"):
        if len(variant) == 1 and len(standard) == 1:
            simplified = chars.get(standard, standard)
            if variant != simplified:
                chars.setdefault(variant, simplified)

    def fold(text: str) -> str:
        return "".join(chars.get(c, c) for c in text)

    phrases: dict[str, str] = {}
    for tw, cn in read("TWPhrasesRev.txt"):
        a, b = fold(tw), fold(cn)
        if a != b and len(a) >= 2 and not any(ch.isascii() for ch in a):
            phrases.setdefault(a, b)
    phrases.update({fold(k): fold(v) for k, v in OVERRIDES_TW.items()})

    # 「簡→繁會改變字形」的簡體字：用來判斷一段文字是不是簡體、需不需要轉換
    # Big5（cp950）收錄的字是繁體常用字（台、游、里…台灣本來就這樣寫），不算簡體專用字
    def in_big5(ch: str) -> bool:
        try:
            ch.encode("cp950")
            return True
        except UnicodeEncodeError:
            return False

    simplified_changeable = sorted({s for s, t in read("STCharacters.txt")
                                    if len(s) == 1 and len(t) == 1 and s != t and not in_big5(s)})

    items = sorted(phrases.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    template = (ROOT / "lingxi" / "hanzi.py").read_text(encoding="utf-8")
    head = template[: template.index("_T = ")]
    tail = template[template.index("_TABLE = "): template.index("PHRASES: dict")]
    rest = template[template.index("_PHRASE_RE = "):]
    body = (
        f"_T = {''.join(chars)!r}\n_S = {''.join(chars.values())!r}\n_SC = {''.join(simplified_changeable)!r}\n{tail}"
        "PHRASES: dict[str, str] = {\n" + ",\n".join(f"    {k!r}: {v!r}" for k, v in items) + ",\n}\n\n" + rest
    )
    (ROOT / "lingxi" / "hanzi.py").write_text(head + body, encoding="utf-8")
    print(f"寫入 lingxi/hanzi.py：{len(chars)} 個單字、{len(phrases)} 組詞彙")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
