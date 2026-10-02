"""Fixed family-first draw plans for the bounded October A/B experiment.

This prepares record IDs, not a Trainer sampler. A caller must freeze and validate
the accepted pools and protected evaluation set before using the plans to train.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .records import CleanupRecord

Pool = Literal["R", "N", "T"]


@dataclass(frozen=True)
class PoolVariant:
    record_id: str
    family_id: str
    language: str
    split: str


@dataclass(frozen=True)
class Draw:
    pool: Pool
    record_id: str
    family_id: str
    language: str


@dataclass(frozen=True)
class ABSchedules:
    a: tuple[Draw, ...]
    b: tuple[Draw, ...]
    seed: int
    real_asr_included: bool


def resolve_draw_plan(
    plan: dict, records: Sequence[CleanupRecord], *, records_sha256: str
) -> tuple[int, ...]:
    """Validate frozen draw metadata before any model load; return ordered indices."""
    if plan.get("schema_version") != "controlled-draw-plan-v1":
        raise ValueError("unsupported draw plan version")
    if plan.get("records_sha256") != records_sha256:
        raise ValueError("draw plan is not bound to these frozen training bytes")
    branch = plan.get("branch")
    if branch not in {"A", "B"} or not isinstance(plan.get("real_asr_included"), bool):
        raise ValueError("draw plan needs branch A/B and an explicit real-ASR flag")
    if any(record.split != "train" for record in records):
        raise ValueError("draw plan accepts only Train records")
    lookup = {record.id: i for i, record in enumerate(records)}
    if len(lookup) != len(records):
        raise ValueError("duplicate training record IDs")
    draws = tuple(Draw(**item) for item in plan["draws"])
    if len(draws) != 1600:
        raise ValueError("controlled runs require exactly 1600 draws")
    indices = []
    families = {}
    for draw in draws:
        if draw.record_id not in lookup:
            raise ValueError("draw references an unknown training record")
        record = records[lookup[draw.record_id]]
        if draw.language != _language(record.language) or not draw.family_id:
            raise ValueError("draw locale/family metadata is invalid")
        family = (record.source_name, record.source_revision,
                  record.source_record_id or record.parent_id or record.id)
        if families.setdefault(family, (draw.pool, draw.family_id)) != (
            draw.pool, draw.family_id
        ):
            raise ValueError("a source family cannot change pool or family identity")
        indices.append(lookup[draw.record_id])
    t = 40 if plan["real_asr_included"] else 0
    expected = Counter()
    for lang in ("de", "en"):
        expected[("R", lang)] = (800 if branch == "A" else 560) - t
        if branch == "B":
            expected[("N", lang)] = 240
        if t:
            expected[("T", lang)] = t
    if Counter((d.pool, d.language) for d in draws) != expected:
        raise ValueError("draw plan does not match the fixed pool/language quotas")
    if branch == "B":
        new_families = set(plan.get("accepted_new_family_ids", []))
        if len(new_families) < 100 or any(
            d.family_id not in new_families for d in draws if d.pool == "N"
        ):
            raise ValueError("B needs a frozen pool of at least 100 accepted new families")
    t_draws = [d for d in draws if d.pool == "T"]
    if t:
        if any(len({d.family_id for d in t_draws if d.language == lang}) < 2
               for lang in ("de", "en")) or len({d.family_id for d in t_draws}) < 5:
            raise ValueError("real-ASR pool is too small for its full allocation")
        if max(Counter(d.family_id for d in t_draws).values()) > 20:
            raise ValueError("real-ASR family exposure exceeds 20")
    return tuple(indices)


def _language(locale: str) -> str:
    if locale == "de-DE":
        return "de"
    if locale in {"en-US", "en-GB"}:
        return "en"
    raise ValueError(f"unsupported main locale: {locale}")


def build_ab_schedules(
    pools: dict[Pool, tuple[PoolVariant, ...]], *, seed: int = 20261002
) -> ABSchedules:
    """Exactly 1,600 draws/branch, 800/language, identical T at identical positions.

    R/N use uniform family selection followed by uniform variant selection. T
    cycles over shuffled families to satisfy its strict exposure cap. There are
    no acceptance decisions here: N must already contain >=100 accepted families.
    """
    if set(pools) != {"R", "N", "T"}:
        raise ValueError("provide the frozen R, N and T pools (T may be empty)")
    grouped = {}
    record_ids: set[str] = set()
    owners: dict[str, str] = {}
    for pool in ("R", "N", "T"):
        by_language = {"de": defaultdict(list), "en": defaultdict(list)}
        for variant in pools[pool]:
            if variant.split != "train":
                raise ValueError("Dev/Test must never enter the draw plan")
            if not variant.record_id or not variant.family_id:
                raise ValueError("record and family IDs must be nonempty")
            if variant.record_id in record_ids:
                raise ValueError("record IDs must be unique across all pools")
            record_ids.add(variant.record_id)
            if owners.setdefault(variant.family_id, pool) != pool:
                raise ValueError("a family cannot belong to multiple pools")
            by_language[_language(variant.language)][variant.family_id].append(variant)
        grouped[pool] = by_language
    if len({v.family_id for v in pools["N"]}) < 100:
        raise ValueError("B requires at least 100 accepted new Train families")
    for pool in ("R", "N"):
        if any(not grouped[pool][lang] for lang in ("de", "en")):
            raise ValueError(f"{pool} must contain both main languages")
    t_de, t_en = set(grouped["T"]["de"]), set(grouped["T"]["en"])
    if t_de & t_en:
        raise ValueError("a real-ASR family must have one documented main language")
    include_t = len(t_de | t_en) >= 5 and min(len(t_de), len(t_en)) >= 2
    rng = random.Random(seed)
    variant_rng = random.Random(seed ^ 0x35)

    def choose(pool: Pool, lang: str, count: int) -> list[Draw]:
        families = grouped[pool][lang]
        keys = sorted(families)
        result = []
        cycle: list[str] = []
        for _ in range(count):
            if pool == "T":
                if not cycle:
                    cycle = keys.copy()
                    rng.shuffle(cycle)
                family = cycle.pop()
            else:
                family = rng.choice(keys)
            variant = variant_rng.choice(sorted(families[family], key=lambda v: v.record_id))
            result.append(Draw(pool, variant.record_id, family, lang))
        return result

    a: list[Draw] = []
    b: list[Draw] = []
    for lang in ("de", "en"):
        t_count = 40 if include_t else 0
        shared_t = choose("T", lang, t_count) if t_count else []
        a_r = iter(choose("R", lang, 800 - t_count))
        b_r = iter(choose("R", lang, 560 - t_count))
        b_n = iter(choose("N", lang, 240))
        slots = ["T"] * t_count + ["R"] * (560 - t_count) + ["N"] * 240
        rng.shuffle(slots)
        t_draws = iter(shared_t)
        for slot in slots:
            if slot == "T":
                draw = next(t_draws)
                a.append(draw)
                b.append(draw)
            else:
                a.append(next(a_r))
                b.append(next(b_r) if slot == "R" else next(b_n))
    order = list(range(1600))
    rng.shuffle(order)
    a_final, b_final = tuple(a[i] for i in order), tuple(b[i] for i in order)
    if max(Counter(d.family_id for d in a_final if d.pool == "T").values(), default=0) > 20:
        raise ValueError("real-ASR exposure exceeded its family cap")
    return ABSchedules(a_final, b_final, seed, include_t)
