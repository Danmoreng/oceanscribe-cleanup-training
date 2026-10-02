from collections import Counter
from dataclasses import asdict

import pytest

from oceanscribe_cleanup.controlled_sampling import (
    PoolVariant,
    build_ab_schedules,
    resolve_draw_plan,
)
from oceanscribe_cleanup.records import CleanupRecord


def pools(t_languages=("de-DE", "de-DE", "de-DE", "en-US", "en-US")):
    result = {}
    for pool, count in (("R", 12), ("N", 100)):
        result[pool] = tuple(
            PoolVariant(f"{pool}-{i}-{j}", f"{pool}-{i}", locale, "train")
            for i in range(count)
            for j, locale in enumerate(("de-DE", "en-US"))
        )
    result["T"] = tuple(
        PoolVariant(f"T-{i}-{j}", f"T-{i}", locale, "train")
        for i, locale in enumerate(t_languages)
        for j in range(2)
    )
    return result


def test_fixed_mix_cap_shared_asr_and_reproducibility():
    inputs = pools()
    plan = build_ab_schedules(inputs)
    assert plan == build_ab_schedules(inputs)
    assert plan.real_asr_included
    assert Counter((d.pool, d.language) for d in plan.a) == {
        ("R", "de"): 760, ("T", "de"): 40, ("R", "en"): 760, ("T", "en"): 40
    }
    assert Counter((d.pool, d.language) for d in plan.b) == {
        ("R", "de"): 520, ("N", "de"): 240, ("T", "de"): 40,
        ("R", "en"): 520, ("N", "en"): 240, ("T", "en"): 40
    }
    for a, b in zip(plan.a, plan.b, strict=True):
        assert a.language == b.language
        if a.pool == "T" or b.pool == "T":
            assert a == b
    assert max(Counter(d.family_id for d in plan.a if d.pool == "T").values()) == 20
    # Variant multiplicity must not change family selection or T exposure.
    inputs["R"] += tuple(
        PoolVariant(f"R-extra-{i}", "R-0", "de-DE", "train") for i in range(40)
    )
    expanded = build_ab_schedules(inputs)
    assert [d.family_id for d in plan.a] == [d.family_id for d in expanded.a]


def test_insufficient_asr_is_omitted_from_both_branches():
    plan = build_ab_schedules(pools(("de-DE", "de-DE", "en-US")))
    assert not plan.real_asr_included
    assert Counter(d.pool for d in plan.a) == {"R": 1600}
    assert Counter(d.pool for d in plan.b) == {"R": 1120, "N": 480}


def test_new_pool_minimum_is_families_not_variants():
    inputs = pools()
    inputs["N"] = inputs["N"][:198]
    with pytest.raises(ValueError, match="100 accepted"):
        build_ab_schedules(inputs)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_protected_splits_rejected(split):
    inputs = pools()
    inputs["T"] += (PoolVariant("protected", "protected", "de-DE", split),)
    with pytest.raises(ValueError, match="Dev/Test"):
        build_ab_schedules(inputs)


def test_cross_pool_family_leakage_rejected():
    inputs = pools()
    inputs["N"] += (PoolVariant("bad-parent", "R-0", "de-DE", "train"),)
    with pytest.raises(ValueError, match="multiple pools"):
        build_ab_schedules(inputs)


def test_frozen_draw_plan_resolves_order_and_rejects_tampering():
    inputs = pools()
    schedule = build_ab_schedules(inputs)
    variants = sum(inputs.values(), ())
    records = tuple(CleanupRecord(
        id=v.record_id, source_name=v.record_id[0], source_record_id=v.family_id,
        language=v.language, split=v.split, transcript=f"Dictation {v.record_id}",
        output=f"Dictation {v.record_id}", synthetic=False,
    ) for v in variants)
    plan = {
        "schema_version": "controlled-draw-plan-v1", "branch": "B",
        "records_sha256": "a" * 64, "real_asr_included": True,
        "accepted_new_family_ids": sorted({v.family_id for v in inputs["N"]}),
        "draws": [asdict(d) for d in schedule.b],
    }
    indices = resolve_draw_plan(plan, records, records_sha256="a" * 64)
    assert [records[i].id for i in indices] == [d.record_id for d in schedule.b]
    with pytest.raises(ValueError, match="frozen training bytes"):
        resolve_draw_plan(plan, records, records_sha256="b" * 64)
    plan["draws"][0]["record_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown training"):
        resolve_draw_plan(plan, records, records_sha256="a" * 64)
