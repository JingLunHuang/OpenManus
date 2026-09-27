// 靈犀 · 頁面快照腳本（注入頁面執行，返回結構化的頁面觀察）
//
// 與 browser-use 的"可互動元素列表"相比，這裡多做了四件事：
// 1. cursor:pointer 啟發式：React/Vue 渲染的 <div> 日曆格子沒有 onclick 屬性，也能被識別；
// 2. 日期格語義：識別 1~31 的日期格，並向上尋找"2026年6月"這類月份標題作為上下文；
// 3. 遮擋檢測：用 elementFromPoint 判斷元素是否被彈窗/浮層蓋住，並找出關閉按鈕；
// 4. 穩定編號：data-lx-id 在元素存活期間保持不變，跨步驟引用 #編號 不會漂移。
(opts) => {
  opts = opts || {};
  const MAX = opts.maxElements || 500;
  const DIGEST = opts.digestChars || 1800;
  const BELOW = opts.belowViewport == null ? 1.0 : opts.belowViewport;

  if (!window.__lx) {
    window.__lx = { seq: 0, mut: 0 };
    try {
      new MutationObserver((ms) => { window.__lx.mut += ms.length; })
        .observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
    } catch (e) { /* 某些頁面禁止觀察，忽略 */ }
  }

  const vw = window.innerWidth, vh = window.innerHeight;
  const TAGS = new Set(["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "SUMMARY", "OPTION"]);
  const ROLES = new Set(["button", "link", "tab", "option", "menuitem", "menuitemradio", "menuitemcheckbox",
    "checkbox", "radio", "combobox", "textbox", "searchbox", "gridcell", "switch", "treeitem", "listbox"]);
  // 比對用詞由 Python 端注入（已展開繁、簡兩種寫法，見 senses/browser.py 的 SNAPSHOT_OPTIONS）
  const CLOSE_WORDS = opts.closeWords || ["×", "✕", "✖", "關閉", "close", "跳過", "我知道了", "稍後", "暫不", "不再提示", "取消"];
  const CHALLENGE = new RegExp(opts.challenge || "安全驗證|人機驗證|captcha|are you a robot|verify you are human", "i");
  const DAY_RE = new RegExp("^(\\d{1,2})\\s*(?:" + (opts.daySuffix || "日|號") + ")?(?:\\s|$|¥|￥)");
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const cut = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const styleCache = new Map();
  const style = (el) => {
    let st = styleCache.get(el);
    if (!st) { st = getComputedStyle(el); styleCache.set(el, st); }
    return st;
  };

  function isCandidate(el) {
    const tag = el.tagName;
    if (TAGS.has(tag)) {
      if (tag === "INPUT" && (el.type === "hidden")) return false;
      if (tag === "A" && !el.hasAttribute("href") && style(el).cursor !== "pointer") return false;
      return true;
    }
    const role = (el.getAttribute("role") || "").toLowerCase();
    if (role && ROLES.has(role)) return true;
    if (el.isContentEditable && el.hasAttribute("contenteditable")) return true;
    if (el.hasAttribute("onclick")) return true;
    const ti = el.getAttribute("tabindex");
    if (ti !== null && Number(ti) >= 0 && tag !== "DIV" && tag !== "BODY") return true;
    if (style(el).cursor === "pointer") {
      const p = el.parentElement;
      if (!p || style(p).cursor !== "pointer") return true; // 只取最外層的 pointer 元素
    }
    return false;
  }

  function visible(el, r) {
    if (r.width < 2 || r.height < 2) return false;
    const st = style(el);
    return st.visibility !== "hidden" && st.display !== "none" && Number(st.opacity) > 0.05;
  }

  function nearbyText(el) {
    let node = el.parentElement;
    const own = clean(el.value || "");
    for (let i = 0; i < 3 && node; i++, node = node.parentElement) {
      const t = clean(node.innerText || "").replace(own, "").trim();
      if (t && t.length <= 24) return t;
      if (t.length > 24) break;
    }
    return "";
  }

  function labelOf(el) {
    const aria = el.getAttribute("aria-label");
    if (aria) return clean(aria);
    const by = el.getAttribute("aria-labelledby");
    if (by) {
      const t = clean(by.split(/\s+/).map((id) => (document.getElementById(id) || {}).innerText || "").join(" "));
      if (t) return t;
    }
    const tag = el.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      if (el.labels && el.labels.length) {
        const t = clean([...el.labels].map((l) => l.innerText).join(" "));
        if (t) return t;
      }
      return clean(el.getAttribute("placeholder") || "") || nearbyText(el) ||
        clean(el.getAttribute("title") || el.getAttribute("name") || "");
    }
    const txt = clean(el.innerText || el.textContent || "");
    if (txt) return txt;
    const img = el.querySelector && el.querySelector("img[alt]");
    return clean(el.getAttribute("title") || el.getAttribute("alt") || (img && img.alt) || el.getAttribute("name") || "");
  }

  function valueOf(el) {
    if (el.tagName === "SELECT") return clean((el.selectedOptions && el.selectedOptions[0] || {}).text || "");
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
      if (el.type === "password") return el.value ? "******" : "";
      return clean(el.value || "");
    }
    if (el.isContentEditable) return clean(el.innerText || "");
    return null;
  }

  const monthCache = new Map();
  const MONTH_RE = /(?:(\d{4})\s*[年\-\/.]\s*)?(\d{1,2})\s*月|([A-Z][a-z]{2,8})\s+(\d{4})/;
  const MONTHS_EN = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
  function monthHeader(node) {
    if (monthCache.has(node)) return monthCache.get(node);
    const head = clean((node.innerText || "").slice(0, 40));
    const m = head.match(MONTH_RE);
    let res = null;
    if (m) {
      if (m[2]) res = { year: m[1] ? Number(m[1]) : null, month: Number(m[2]) };
      else if (m[3]) {
        const idx = MONTHS_EN.indexOf(m[3].slice(0, 3).toLowerCase());
        if (idx >= 0) res = { year: Number(m[4]), month: idx + 1 };
      }
    }
    monthCache.set(node, res);
    return res;
  }

  const FULL_DATE = /(\d{4})[-\/年.](\d{1,2})[-\/月.](\d{1,2})/;
  function dateInfo(el, label) {
    const first = clean((el.innerText || label || "").split("\n")[0]);
    const m = first.match(DAY_RE) || label.match(/^(\d{1,2})(?:\s|$)/);
    let iso = null;
    for (const attr of ["data-date", "data-day", "data-value", "date", "aria-label", "title"]) {
      const v = el.getAttribute(attr);
      const mm = v && v.match(FULL_DATE);
      if (mm) { iso = `${mm[1]}-${mm[2].padStart(2, "0")}-${mm[3].padStart(2, "0")}`; break; }
    }
    if (!m && !iso) return null;
    const day = iso ? Number(iso.slice(8, 10)) : Number(m[1]);
    if (day < 1 || day > 31) return null;
    const calClass = el.closest('[class*="calendar" i],[class*="date" i],[class*="picker" i],[class*="month" i],[role="grid"]');
    let month = null, node = el.parentElement;
    for (let i = 0; i < 8 && node && !month; i++, node = node.parentElement) month = monthHeader(node);
    // 普通表格裡的數字不算日期：必須有日曆類名、完整日期屬性，或"表格 + 月份標題"
    if (!iso && !calClass && !(el.closest("table") && month)) return null;
    const cls = (el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className) || "";
    const disabled = /disable|invalid|past|gray/i.test(cls) || el.getAttribute("aria-disabled") === "true";
    return { day, iso, year: month && month.year, month: month && month.month, disabled };
  }

  function kindOf(el, label, date) {
    const tag = el.tagName, role = (el.getAttribute("role") || "").toLowerCase(), type = (el.type || "").toLowerCase();
    if (date) return "date";
    if (tag === "INPUT" && (type === "checkbox" || type === "radio")) return "check";
    if (role === "checkbox" || role === "radio" || role === "switch") return "check";
    if (tag === "INPUT" && ["submit", "button", "reset", "image"].includes(type)) return "button";
    if (tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable || ["textbox", "searchbox", "combobox"].includes(role)) return "input";
    if (tag === "SELECT" || role === "listbox") return "select";
    if (role === "tab") return "tab";
    if (role === "option" || role === "menuitem" || tag === "OPTION" || el.closest('[role="listbox"],[class*="suggest" i],[class*="dropdown" i],[class*="associat" i]')) return "option";
    if (tag === "A" && el.hasAttribute("href")) return "link";
    if (tag === "BUTTON" || role === "button") return "button";
    if (label && label.length <= 12) return "button";
    return "other";
  }

  function statesOf(el) {
    const s = [];
    if (el.disabled || el.getAttribute("aria-disabled") === "true") s.push("disabled");
    if (el.checked || el.getAttribute("aria-checked") === "true") s.push("checked");
    if (el.getAttribute("aria-selected") === "true" || /\b(active|selected|current|cur|on)\b/i.test(typeof el.className === "string" ? el.className : "")) s.push("selected");
    if (el.getAttribute("aria-expanded") === "true") s.push("expanded");
    if (document.activeElement === el) s.push("focused");
    return s;
  }

  // 找到遮擋物所在的"圖層"：最外層的 fixed / 帶 z-index 的 absolute 祖先
  function layerOf(node) {
    let best = node, n = node;
    while (n && n !== document.body && n !== document.documentElement) {
      const st = style(n);
      if (st.position === "fixed" || (st.position === "absolute" && st.zIndex !== "auto")) best = n;
      n = n.parentElement;
    }
    return best;
  }

  function covering(el, r) {
    const cx = Math.min(Math.max(r.left + r.width / 2, 0), vw - 1);
    const cy = Math.min(Math.max(r.top + r.height / 2, 0), vh - 1);
    const hit = document.elementFromPoint(cx, cy);
    if (!hit || hit === el || el.contains(hit) || hit.contains(el)) return null;
    return hit;
  }

  // ---------- 遍歷（含開放的 Shadow DOM）----------
  const found = [];
  function collect(root) {
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    let n;
    while ((n = tw.nextNode())) {
      if (isCandidate(n)) found.push(n);
      if (n.shadowRoot) collect(n.shadowRoot);
    }
  }
  collect(document.body || document.documentElement);

  const coverCount = new Map();
  const items = [];
  for (const el of found) {
    const r = el.getBoundingClientRect();
    if (!visible(el, r)) continue;
    if (r.bottom < 0 || r.top > vh * (1 + BELOW) || r.right < 0 || r.left > vw) continue;
    const inView = r.top < vh && r.bottom > 0;
    const label = cut(labelOf(el), 80);
    const date = dateInfo(el, label);
    const kind = kindOf(el, label, date);
    if (kind === "other" && !label) continue;
    let id = el.getAttribute("data-lx-id");
    if (!id) { id = String(++window.__lx.seq); el.setAttribute("data-lx-id", id); }
    const cover = inView ? covering(el, r) : null;
    if (cover) {
      const key = layerOf(cover);
      coverCount.set(key, (coverCount.get(key) || 0) + 1);
    }
    const item = {
      id: Number(id), tag: el.tagName.toLowerCase(), kind, label,
      value: valueOf(el), role: el.getAttribute("role") || "",
      type: (el.type || "").toLowerCase(),
      href: el.tagName === "A" ? cut(el.getAttribute("href") || "", 80) : "",
      rect: { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) },
      inView, covered: !!cover, states: statesOf(el),
    };
    if (date) item.date = date;
    items.push(item);
  }

  // 數量過多時優先保留視口內的元素
  let elements = items;
  if (items.length > MAX) {
    const inside = items.filter((i) => i.inView), outside = items.filter((i) => !i.inView);
    elements = inside.concat(outside).slice(0, MAX);
  }

  // ---------- 浮層 / 彈窗 ----------
  let overlay = null;
  let topCover = null, topCount = 0;
  coverCount.forEach((c, node) => { if (c > topCount) { topCount = c; topCover = node; } });
  if (topCover && topCount >= 3) {
    const r = topCover.getBoundingClientRect();
    const closeIds = elements
      .filter((i) => !i.covered && i.inView && CLOSE_WORDS.some((w) => i.label.toLowerCase().includes(w.toLowerCase()) && i.label.length <= 8))
      .map((i) => i.id);
    overlay = {
      coveredCount: topCount,
      area: Math.round((r.width * r.height) / (vw * vh) * 100),
      hint: cut(clean(topCover.innerText || ""), 60),
      closeIds,
    };
  }

  // ---------- 正文摘要與攔截檢測 ----------
  const root = document.querySelector("main,[role=main],article") || document.body;
  const text = (root ? root.innerText || "" : "").split("\n").map(clean).filter(Boolean).join("\n");
  const bodyLen = clean(document.body ? document.body.innerText : "").length;
  // 注意不能用裸的"驗證碼"：登入框裡的"獲取驗證碼"（簡訊）不是反爬挑戰
  const challenge = CHALLENGE.test(text.slice(0, 3000) + " " + document.title);

  return {
    url: location.href,
    title: document.title,
    viewport: { w: vw, h: vh, dpr: window.devicePixelRatio || 1 },
    scroll: {
      y: Math.round(window.scrollY),
      height: Math.round(document.documentElement.scrollHeight),
      below: Math.max(0, Math.round(document.documentElement.scrollHeight - window.scrollY - vh)),
    },
    elements,
    overlay,
    focusedId: document.activeElement && document.activeElement.getAttribute ? Number(document.activeElement.getAttribute("data-lx-id")) || null : null,
    digest: text.slice(0, DIGEST),
    textLength: text.length,
    iframes: document.querySelectorAll("iframe").length,
    suspectBlocked: bodyLen < 40 && elements.length < 3,
    challenge,
    mutations: window.__lx.mut,
  };
}
