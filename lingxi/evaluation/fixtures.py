"""建立離線實驗與受保護評測共用的本地網頁。"""

from __future__ import annotations

from pathlib import Path

from lingxi.hanzi import unify


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_site_dir(target: Path) -> Path:
    """建立繁體／簡體航班頁與長文頁。"""
    target.mkdir(parents=True, exist_ok=True)
    html = (FIXTURES / "flight.html").read_text(encoding="utf-8")
    (target / "flight.html").write_text(html, encoding="utf-8")
    cn = unify(html).replace('lang="zh-Hant"', 'lang="zh-CN"')
    (target / "flight-cn.html").write_text(cn, encoding="utf-8")
    filler = "<p>" + "本站提供國內外機票、酒店預訂服務，客服電話全天候為您服務。" * 8 + "</p>"
    key = "<p>退改簽規則：起飛前 24 小時以上免費退票，24 小時內收取票面價 20% 的手續費。</p>"
    article = ("<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>乘機須知</title></head><body>"
               + filler * 55 + key + filler * 10 + "</body></html>")
    (target / "long.html").write_text(article, encoding="utf-8")
    return target
