"""LightMem 寫入管線（Fang et al.，ICLR 2026，arXiv:2510.18866，github.com/zjunlp/LightMem）
套用到「代理的執行經驗」上。

論文仿照 Atkinson–Shiffrin 人類記憶模型，把寫入拆成三段；靈犀的對應做法：

  感官記憶（預壓縮）
    論文：LLMLingua-2 逐 token 打保留分，動態門檻 τ = 第 r 百分位，只留分數高於 τ 的 token（r≈0.6）
    靈犀：以「子句」為單位打資訊密度分 ＝ 顯著度（數字/日期/價格、經驗線索詞、失敗訊號）＋ 新穎度（與先前子句的最大相似度取反），
          同樣用動態門檻保留前 r 比例。不需要下載 BERT 模型，離線可用。
  主題分段
    論文：注意力邊界 B1 ∩ 語義相似度邊界 B2（兩者都成立才切，準確率 > 80%）
    靈犀：B1 = 情境轉換（換站點 / 換工具族），B2 = 相鄰兩步的內容相似度 < τ，同樣取交集
  短期記憶
    論文：按主題分組緩衝，累積到 th token 才呼叫一次摘要，而不是每輪都寫
    靈犀：同上；執行中零模型呼叫——線上摘要是抽取式，睡眠時段再交給模型重塑
  長期記憶
    論文：測試時軟更新（只插入、帶時間戳）＋ 睡眠時段離線更新：每個條目 e_i 的更新佇列
          Q(e_i) = 比它新（t_j ≥ t_i）且相似度夠高的前 k 個條目；各佇列互不相依，可平行處理
    靈犀：同上。網頁事實會過期（價格會變）：同一主題的新值取代舊值，舊值標為 superseded 並保留歷史
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from lingxi.hanzi import unify
from lingxi.journal.recorder import Event
from lingxi.llm.messages import Message
from lingxi.memory.graph import MemNode, MemoryGraph
from lingxi.retrieval import cosine, estimate_tokens
from lingxi.retrieval.embed import max_previous_similarity
from lingxi.retrieval.text import sentences
from lingxi.utils import clip, extract_json, site_key

_DIGIT = re.compile(r"\d")
_CUE = re.compile("必須|需要|要先|不要|別再|改用|才會|才能|無法|不能|失敗|注意|攔截|驗證碼|登入|浮層|遮擋|聯想|候選|日曆|直達|"
                  "等待|載入|反爬|彈窗|只能|應該|記得|避免|改成|其實|原來|直接")
_FAIL = re.compile("✘|失敗|未驗證|錯誤|超時|找不到|被遮擋|退回")
_ELEMENT_ID = re.compile(r"#\d+\s*")
_NUMBERS = re.compile(r"\d+(?:[.:]\d+)?")
_CLOSE = ("×", "x", "X", "✕", "close")


# ---------------------------------------------------------------- 執行摘錄
@dataclass
class Action:
    name: str
    args: dict[str, Any]
    ok: bool = True
    verified: bool | None = None
    summary: str = ""
    detail: str = ""
    sig: str = ""


@dataclass
class Step:
    step: int
    thought: str = ""
    actions: list[Action] = field(default_factory=list)
    site: str = ""
    notes: list[str] = field(default_factory=list)

    def family(self) -> str:
        names = {a.name for a in self.actions}
        if any(n.startswith("web_") for n in names):
            return "web"
        for fam in ("python_run", "files", "web_search", "recall"):
            if fam in names:
                return fam
        return "think"


@dataclass
class RunDigest:
    run_id: str = ""
    task: str = ""
    profile: str = ""
    status: str = ""
    answer: str = ""
    started: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    playbooks: list[str] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    verify_rounds: int = 0

    def main_site(self) -> str:
        sites = Counter(s.site for s in self.steps if s.site)
        return sites.most_common(1)[0][0] if sites else ""


def signature(a: Action, grounding: dict | None, prev_typed: str) -> str:
    """把一個動作抽象成可跨任務比對的簽名：去掉城市、日期、元素編號這類隨任務變化的值。"""
    n, args = a.name, a.args
    if n == "web_open":
        u = urlparse(str(args.get("url", "")))
        seg = next((p for p in u.path.split("/") if p and not _DIGIT.search(p) and "." not in p), "")
        return f"web_open({site_key(u.geturl())}{'/' + seg if seg else ''})"
    if n == "web_click":
        target = str(args.get("target", "")).strip()
        if (grounding or {}).get("method") == "date":
            return "web_click(〈日期格〉)"
        if target in _CLOSE or unify("關閉") in unify(target):  # 繁簡寫法都算關閉按鈕
            return "web_click(〈關閉浮層〉)"
        if prev_typed and prev_typed in target:
            return "web_click(〈聯想候選〉)"
        if target.startswith("#") or _DIGIT.search(target):
            return "web_click(〈編號元素〉)"
        return f"web_click({clip(target, 12)})"
    if n in ("web_type", "web_select"):
        return f"{n}({clip(str(args.get('target', '')), 12)})"
    if n in ("files", "plan"):
        return f"{n}({args.get('action', '')})"
    return n


def digest_run(events: list[Event], status: str | None = None, answer: str | None = None,
               run_id: str = "") -> RunDigest:
    d = RunDigest(run_id=run_id)
    if events:
        d.started = events[0].ts
    steps: dict[int, Step] = {}
    site = ""
    prev_typed = ""
    pending_calls: list[dict] = []
    for ev in events:
        data = ev.data
        k = ev.kind
        if k == "run.start":
            d.task = data.get("task", "")
        elif k == "intent":
            d.profile = data.get("profile", "")
        elif k == "temporal":
            d.anchors = [a.get("date", "") for a in data.get("anchors", [])]
        elif k == "playbook":
            d.playbooks.append(data.get("id", ""))
        elif k == "page":
            site = site_key(data.get("url", "")) or site
        elif k == "think":
            st = steps.setdefault(ev.step, Step(ev.step, site=site))
            st.thought = data.get("content", "") or ""
            pending_calls = list(data.get("calls", []))
        elif k == "outcome":
            st = steps.setdefault(ev.step, Step(ev.step, site=site))
            call = pending_calls.pop(0) if pending_calls else {"name": data.get("name"), "args": {}}
            a = Action(name=data.get("name", ""), args=call.get("args", {}) or {}, ok=bool(data.get("ok")),
                       verified=data.get("verified"), summary=data.get("summary", ""), detail=data.get("detail", ""))
            if a.name == "web_open" and a.ok:
                site = site_key(str(a.args.get("url", ""))) or site
            jumped = re.search(r"頁面跳轉到 (\S+)", a.summary)
            if jumped:
                site = site_key(jumped.group(1)) or site
            a.sig = signature(a, data.get("grounding"), prev_typed)
            prev_typed = str(a.args.get("text", "")) if a.name == "web_type" else ""
            st.actions.append(a)
            st.site = st.site or site
        elif k in ("guard", "error"):
            steps.setdefault(ev.step, Step(ev.step, site=site)).notes.append(data.get("message", ""))
        elif k == "verify":
            d.verify_rounds = max(d.verify_rounds, int(data.get("round", 0) or 0))
            if data.get("action") == "reject":
                bad = [c.get("text", "") for c in data.get("claims", []) if c.get("verdict") != "supported"]
                steps.setdefault(ev.step, Step(ev.step, site=site)).notes.append("查證退回：無據陳述「" + "、".join(bad) + "」")
        elif k == "findings":
            d.findings = data.get("items", [])
        elif k == "run.finish":
            d.status = data.get("status", d.status)
            d.answer = data.get("answer", d.answer)
    d.steps = [steps[i] for i in sorted(steps)]
    if status is not None:
        d.status = status
    if answer is not None:
        d.answer = answer
    return d


# ---------------------------------------------------------------- 感官記憶：預壓縮
def step_clauses(st: Step) -> list[tuple[str, str]]:
    """一步拆成 (來源, 子句)：模型的想法、動作與結果、警告與查證。"""
    out: list[tuple[str, str]] = [("thought", s) for s in sentences(st.thought)]
    for a in st.actions:
        mark = "✔" if a.ok else "✘"
        if a.verified is False:
            mark += "（未驗證）"
        out.append(("action", f"{a.sig} {mark} {clip(a.summary, 100)}"))
        for line in (a.detail or "").splitlines()[:2]:
            if line.strip() and a.name not in ("finish", "web_read"):
                out.append(("detail", clip(line.strip(), 120)))
    out += [("note", n) for n in st.notes if n]
    return out


class SensoryCompressor:
    def __init__(self, embedder, ratio: float = 0.6):
        self.embedder = embedder
        self.ratio = ratio

    @staticmethod
    def salience(source: str, text: str) -> float:
        score = min(len(text) / 40, 1.0) * 0.3
        score += 0.6 if _DIGIT.search(text) else 0.0
        score += 1.0 if _CUE.search(text) else 0.0
        score += 0.8 if _FAIL.search(text) else 0.0
        score += {"thought": 0.5, "note": 0.7}.get(source, 0.0)  # 模型的想法、警告比工具回傳更接近「經驗」
        return score

    def compress(self, items: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[float]]:
        if not items:
            return [], []
        vecs = self.embedder.embed([t for _, t in items])
        prev_max = max_previous_similarity(vecs)
        scores = [self.salience(source, text) + max(1.0 - prev_max[i], 0.0) for i, (source, text) in enumerate(items)]
        keep_n = max(1, round(len(items) * self.ratio))
        # 動態門檻 τ：保留分數最高的前 r 比例，再按原順序排回去
        top = sorted(sorted(range(len(items)), key=lambda i: -scores[i])[:keep_n])
        return [items[i] for i in top], [scores[i] for i in top]


# ---------------------------------------------------------------- 主題分段 ＋ 短期記憶
def segment(steps: list[Step], vecs: list[list[float]], threshold: float) -> list[list[int]]:
    if not steps:
        return []
    groups = [[0]]
    for k in range(1, len(steps)):
        prev, cur = steps[k - 1], steps[k]
        b1 = (cur.site and prev.site and cur.site != prev.site) or cur.family() != prev.family()
        b2 = cosine(vecs[k], vecs[k - 1]) < threshold
        if b1 and b2:
            groups.append([k])
        else:
            groups[-1].append(k)
    return groups


class ShortTermMemory:
    """按主題（站點或工具族）分組的緩衝；某個主題累積到 th token 才交出去摘要一次。"""

    def __init__(self, threshold_tokens: int):
        self.th = threshold_tokens
        self.buffers: dict[str, list[tuple[str, str]]] = {}

    def push(self, topic: str, items: list[tuple[str, str]]) -> list[tuple[str, list[tuple[str, str]]]]:
        buf = self.buffers.setdefault(topic, [])
        buf.extend(items)
        if sum(estimate_tokens(t) for _, t in buf) >= self.th:
            return [(topic, self.buffers.pop(topic))]
        return []

    def flush_all(self) -> list[tuple[str, list[tuple[str, str]]]]:
        out = list(self.buffers.items())
        self.buffers = {}
        return out


# 記憶投毒防護：網頁內容會被寫進長期記憶、之後再放進系統提示，
# 等於給了惡意網頁一條「持久化提示注入」的路徑（參考 zjunlp/LightMem issue #86 的記憶投毒揭露）。
# 看起來像在對代理下指令的內容一律不寫入。
_INJECTION = re.compile(
    r"忽略(之前|以上|先前|前面|所有)的?(指令|指示|規則|設定)|無視(之前|以上|所有)|"
    r"(系統|system)\s*(提示|prompt)|你(現在)?(必須|一定要)|從現在(開始|起)你|"
    r"ignore\s+(all\s+|the\s+|any\s+)?(previous|prior|above)|disregard\s+(all|previous|the)|"
    r"you\s+(must|are\s+now)|</?\s*(system|assistant|instruction)", re.I)


def looks_like_instruction(text: str) -> bool:
    return bool(_INJECTION.search(unify(text)) or _INJECTION.search(text))


_ENV_CUE = re.compile("浮層|遮擋|彈窗|登入|驗證碼|攔截|反爬|聯想|候選|日曆|日期格|下拉|按鈕|輸入框|直達|載入|非同步|"
                      "跳轉|分頁|捲動|點選|必須|才會|才能|要先|改用|不要|無法|只能")
_SPECIFIC = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}[日號]|[¥$￥]\s*\d|「[^」]*\d[^」]*」")
_ANSWER_TALK = re.compile("查證|答覆|依據|刪除|退回|finish|發現板")


def note_score(text: str) -> float:
    """「可重用的站點經驗」分數：描述網站環境與操作規律的加分；寫死日期、價格，或只是在談答覆與查證的扣分。"""
    return (len(_ENV_CUE.findall(text)) * 1.0 - (0.8 if _SPECIFIC.search(text) else 0.0)
            - (1.5 if _ANSWER_TALK.search(text) else 0.0))


def extract_notes(items: list[tuple[str, str]], embedder, max_notes: int = 3) -> list[str]:
    """抽取式摘要：從模型的想法裡挑出最像「可重用站點經驗」的句子（線上寫入用，零模型呼叫）。"""
    cands = []
    for source, text in items:
        if source != "thought":
            continue
        clean = _ELEMENT_ID.sub("", text).strip(" 。，")
        score = note_score(clean)
        if len(clean) < 6 or score < 1.0:
            continue
        cands.append((score, clean))
    cands.sort(key=lambda x: -x[0])
    notes: list[str] = []
    vecs: list[list[float]] = []
    for _, text in cands:
        v = embedder.embed([text])[0]
        if any(cosine(v, o) > 0.85 for o in vecs):
            continue
        notes.append(text)
        vecs.append(v)
        if len(notes) >= max_notes:
            break
    return notes


# ---------------------------------------------------------------- 線上寫入（軟更新）
class LightMemWriter:
    def __init__(self, graph: MemoryGraph, cfg):
        self.graph = graph
        self.cfg = cfg

    def write(self, d: RunDigest) -> dict[str, Any]:
        g = self.graph
        compressor = SensoryCompressor(g.embedder, self.cfg.compress_ratio)
        raw_items = [step_clauses(st) for st in d.steps]
        flat = [it for items in raw_items for it in items]
        kept_items, _ = compressor.compress(flat)
        kept_set = {id(it) for it in kept_items}
        per_step = [[it for it in items if id(it) in kept_set] for items in raw_items]

        step_vecs = g.embedder.embed([" ".join(t for _, t in items) or st.thought or "—"
                                      for st, items in zip(d.steps, per_step)])
        groups = segment(d.steps, step_vecs, self.cfg.segment_threshold)
        stm = ShortTermMemory(self.cfg.stm_tokens)
        flushed = []
        for group in groups:
            topic = d.steps[group[0]].site or d.steps[group[0]].family()
            items = [it for k in group for it in per_step[k]]
            flushed += stm.push(topic, items)
        flushed += stm.flush_all()

        now = d.started  # 記憶的時間戳是「經歷發生的時間」，補寫歷史執行時離線更新才能分清新舊
        episode = g.add(MemNode(
            id="", layer="episodic", title=d.task, site=d.main_site(), run_id=d.run_id, created=d.started,
            text="\n".join(t for _, t in kept_items),
            data={"task": d.task, "profile": d.profile, "status": d.status, "answer": clip(d.answer, 400),
                  "playbooks": d.playbooks, "anchors": d.anchors, "verify_rounds": d.verify_rounds,
                  "steps": [{"a": a.sig, "ok": a.ok, "v": a.verified, "site": st.site}
                            for st in d.steps for a in st.actions]}))
        note_ids, fact_ids, blocked = [], [], 0
        for topic, items in flushed:
            site = topic if "." in topic or topic.replace(".", "").isdigit() else ""
            for text in extract_notes(items, g.embedder):
                if looks_like_instruction(text):
                    blocked += 1
                    continue
                node = g.add(MemNode(id="", layer="semantic", text=text, site=site, run_id=d.run_id, created=now,
                                     sources=[d.run_id] if d.run_id else [],
                                     data={"kind": "note", "support": 1, "origin": "agent"}))
                g.link("ground", node.id, episode.id)
                note_ids.append(node.id)
        if d.status in ("success", "partial"):
            for f in d.findings:
                if f.get("status") == "conflict" or not f.get("text"):
                    continue
                if looks_like_instruction(f["text"]):  # 來自網頁的「事實」夾帶指令：不寫入
                    blocked += 1
                    continue
                srcs = list(f.get("sources", []))
                node = g.add(MemNode(id="", layer="semantic", text=f["text"], site=site_key(srcs[0]) if srcs else "",
                                     run_id=d.run_id, created=now, sources=srcs,
                                     data={"kind": "fact", "support": max(1, len(srcs)), "status": f.get("status"),
                                           "anchors": d.anchors, "task": d.task, "origin": "web"}))
                g.link("ground", node.id, episode.id)
                fact_ids.append(node.id)
        return {"episode": episode.id, "notes": note_ids, "facts": fact_ids, "segments": len(groups),
                "flushes": len(flushed), "clauses": len(flat), "kept": len(kept_items), "blocked": blocked,
                "ratio": round(len(kept_items) / len(flat), 3) if flat else 0.0}


# ---------------------------------------------------------------- 睡眠時段：離線更新
_UPDATE_PROMPT = """你在整理代理的長期記憶。下面是一條舊記憶和幾條比它新、內容相近的記憶。
請判斷舊記憶該如何處理，只輸出 JSON：{{"action": "merge" | "update" | "keep", "target": "新記憶編號", "text": "合併後的文字（merge 時必填，一律繁體中文）"}}
- merge：說的是同一件事（可合併成一條，保留所有關鍵細節）
- update：同一主題但新記憶的說法已經變了（例如價格、規則改了），舊的應被新的取代
- keep：其實是不同的事，兩條都保留

舊記憶 {old_id}（{old_age}）：{old_text}
較新的記憶：
{cands}"""


def _rule_decide(old: MemNode, new: MemNode, sim: float) -> str:
    if old.data.get("kind") == "fact":
        if old.data.get("anchors") != new.data.get("anchors"):
            return "keep"  # 不同日期的查詢結果不是同一件事
        if _NUMBERS.findall(old.text) == _NUMBERS.findall(new.text):
            return "merge"
        if old.data.get("support", 1) > new.data.get("support", 1):
            # 防投毒：多方印證過的舊事實，不能被單一新來源直接取代；兩條都保留並記下矛盾，交給下次查證
            old.data.setdefault("conflicts", []).append({"id": new.id, "text": new.text})
            return "keep"
        return "update"
    return "merge" if sim >= 0.9 else "keep"


async def offline_update(graph: MemoryGraph, llm=None, queue_k: int = 3, threshold: float = 0.8,
                         concurrency: int = 4) -> dict[str, Any]:
    entries = sorted(graph.active("semantic"), key=lambda n: n.created)
    queues: dict[str, list[tuple[MemNode, float]]] = {}
    for e in entries:
        hits = graph.index.search(e.text, k=queue_k * 4, where={"layer": "semantic"})
        cands = [(graph.nodes[h.doc.id], h.dense) for h in hits if h.doc.id != e.id]
        cands = [(n, s) for n, s in cands if n.active and n.created >= e.created and s >= threshold
                 and n.data.get("kind") == e.data.get("kind")]
        if cands:
            queues[e.id] = sorted(cands, key=lambda x: -x[1])[:queue_k]

    sem = asyncio.Semaphore(concurrency)

    async def decide(old_id: str) -> tuple[str, str, str, str]:
        old = graph.nodes[old_id]
        cands = queues[old_id]
        if llm is None:
            # 規則模式依相似度走完整個佇列，取第一個不是 keep 的決定
            # （最相似的可能只是「不同日期的同一句話」，真正的更新排在後面）
            for new, sim in cands:
                action = _rule_decide(old, new, sim)
                if action != "keep":
                    return old_id, action, new.id, ""
            return old_id, "keep", cands[0][0].id, ""
        async with sem:
            lines = "\n".join(f"{n.id}（{n.age()}）：{n.text}" for n, _ in cands)
            reply = await llm.chat([Message.user(_UPDATE_PROMPT.format(old_id=old.id, old_age=old.age(),
                                                                       old_text=old.text, cands=lines))])
        data = extract_json(reply.content) or {}
        target = str(data.get("target") or cands[0][0].id)
        return old_id, str(data.get("action", "keep")), target, str(data.get("text", ""))

    decisions = await asyncio.gather(*(decide(i) for i in queues))  # 各佇列互相獨立，平行處理
    merged = updated = 0
    for old_id, action, target_id, text in sorted(decisions, key=lambda x: graph.nodes[x[0]].created):
        old, target = graph.nodes[old_id], graph.nodes.get(target_id)
        while target is not None and not target.active and target.superseded_by:
            target = graph.nodes.get(target.superseded_by)  # 目標自己也被取代了：沿著取代鏈找最新的
        if not old.active or target is None or not target.active or target.id == old.id:
            continue
        if action == "merge":
            target.sources = list(dict.fromkeys(target.sources + old.sources))
            target.data["support"] = target.data.get("support", 1) + old.data.get("support", 1)
            if text:
                target.text = text
            merged += 1
        elif action == "update":
            target.data.setdefault("history", []).append({"text": old.text, "created": old.created, "id": old.id})
            updated += 1
        else:
            continue
        graph.supersede(old, target)
        graph.touch(target)
    return {"queues": len(queues), "merged": merged, "updated": updated}
