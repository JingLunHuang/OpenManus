"""Capture the text actually read by the agent and validate short quotations.

The journal keeps short quotations by default. Full page text lives only for
the duration of the run unless ``save_full_text`` is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable
from urllib.parse import urlparse

from lingxi.hanzi import unify
from lingxi.utils import site_key

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_SPACE = re.compile(r"\s+")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check_quote(text: str, claim: str, quote: str, subject: str = "",
                effective_date: str = "") -> int | None:
    """Return the raw-text offset when a quotation supports the stated facets.

    This checks literal evidence, not the truth of a paraphrase. Numeric claims
    with no declared subject need a close literal match to avoid accepting the
    same number for a different entity elsewhere on the page.
    """
    quote = quote.strip()
    if not quote or len(quote) > 500:
        return None
    offset = text.find(quote)
    if offset < 0:
        return None
    claim_numbers = _numbers(claim)
    if not claim_numbers <= _numbers(quote):
        return None
    normalized_quote = unify(quote)
    if subject and (unify(subject) not in normalized_quote or unify(subject) not in unify(claim)):
        return None
    if effective_date and unify(effective_date) not in normalized_quote:
        return None
    if subject and claim_numbers:
        def relation_bigrams(value: str) -> set[str]:
            residual = unify(value).replace(unify(subject), "")
            residual = _NUMBER.sub("", residual)
            return {piece[i:i + 2] for piece in re.findall(r"[\u4e00-\u9fff]{2,}", residual)
                    for i in range(len(piece) - 1)}

        wanted = relation_bigrams(claim)
        found = relation_bigrams(quote)
        if wanted and len(wanted & found) / len(wanted) < 0.33:
            return None
    if claim_numbers and not subject:
        left = _SPACE.sub("", unify(claim))
        right = _SPACE.sub("", normalized_quote)
        if left not in right:
            return None
    if not claim_numbers:
        # Without a numeric anchor, accepting any quotation from the page would
        # let an unrelated sentence validate a model-written claim. Require the
        # short claim itself to occur in the quoted text.
        left = _SPACE.sub("", unify(claim))
        right = _SPACE.sub("", normalized_quote)
        if left not in right:
            return None
    return offset


@dataclass(frozen=True)
class SourceSnapshot:
    id: str
    url: str
    canonical_url: str
    captured_at: float
    text_hash: str
    text: str
    artifact: str | None = None
    publisher: str = ""


def evidence_state(proofs: list[dict]) -> str:
    valid = [p for p in proofs if p.get("valid", True) and p.get("group")]
    if not valid:
        return "unverified"
    groups = {p["group"] for p in valid}
    publishers = {group for group in groups if group.startswith("publisher:")}
    if len(publishers) >= 2:
        return "independently_supported"
    return "independence_unknown" if len(groups) >= 2 else "single_supported"


def independent_group(candidate: SourceSnapshot, prior: list[tuple[SourceSnapshot, str]]) -> str:
    """Return a conservative publisher group, reusing one for likely copies."""
    candidate_site = site_key(candidate.url)
    candidate_canonical = candidate.canonical_url or candidate.url
    for existing, group in prior:
        existing_canonical = existing.canonical_url or existing.url
        if candidate_site == site_key(existing.url) or candidate_canonical == existing_canonical:
            return group
        if candidate.text_hash == existing.text_hash:
            return group
        a = _SPACE.sub(" ", candidate.text).strip()[:5000]
        b = _SPACE.sub(" ", existing.text).strip()[:5000]
        if a and b and SequenceMatcher(None, a, b).ratio() >= 0.88:
            return group
    if candidate.publisher.strip():
        return "publisher:" + _SPACE.sub(" ", unify(candidate.publisher)).strip().casefold()
    return "unknown:" + (candidate_site or urlparse(candidate.url).netloc or candidate.url)


class EvidenceLedger:
    def __init__(self, emit: Callable, save_artifact: Callable, *, save_full_text: bool = False):
        self.emit = emit
        self.save_artifact = save_artifact
        self.save_full_text = save_full_text
        self.snapshots: dict[str, SourceSnapshot] = {}
        self._span_count = 0

    def capture(self, url: str, text: str, canonical_url: str = "", publisher: str = "") -> SourceSnapshot:
        snapshot_id = f"S{len(self.snapshots) + 1}"
        artifact = self.save_artifact(f"evidence_{snapshot_id}.txt", text) if self.save_full_text else None
        snap = SourceSnapshot(snapshot_id, url, canonical_url, time.time(), _hash(text), text, artifact, publisher)
        self.snapshots[snapshot_id] = snap
        self.emit("evidence.snapshot", schema=1, snapshot_id=snapshot_id, url=url,
                  canonical_url=canonical_url, captured_at=snap.captured_at,
                  text_hash=snap.text_hash, chars=len(text), artifact=artifact, publisher=publisher)
        return snap

    def prove(self, snapshot: SourceSnapshot, claim: str, quote: str, *,
              subject: str = "", effective_date: str = "",
              prior: list[dict] | None = None) -> dict | None:
        offset = check_quote(snapshot.text, claim, quote, subject, effective_date)
        if offset is None:
            return None
        self._span_count += 1
        span_id = f"P{self._span_count}"
        prior_snapshots = [(self.snapshots[p["snapshot_id"]], p["group"])
                           for p in (prior or []) if p.get("snapshot_id") in self.snapshots]
        group = independent_group(snapshot, prior_snapshots)
        proof = {"snapshot_id": snapshot.id, "span_id": span_id, "url": snapshot.url,
                 "quote": quote.strip(), "quote_hash": _hash(quote.strip()),
                 "group": group, "offset": offset, "captured_at": snapshot.captured_at}
        self.emit("evidence.span", schema=1, **proof)
        return proof
