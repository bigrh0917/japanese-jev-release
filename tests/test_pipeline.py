"""Assert-style checks with no dependency on pytest, torch or datasets:

    python tests/test_pipeline.py

They cover the places that would break silently: the template text matches the agreed format character for
character; every adapter has the fields the registry requires; no config under configs/ puts an evaluation-only
dataset into the training sources; relative paths resolve to the project root, not to configs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from japanese_jev import config as cfgmod  # noqa: E402
from japanese_jev.data_pipeline import adapters  # noqa: E402
from japanese_jev.data_pipeline.build import check_sources  # noqa: E402
from japanese_jev.template import KINDS, LETTERS, render  # noqa: E402


def test_render_matches_format():
    got = render("choice",
                 "I was charged twice for my flight to Berlin and nobody answers the phone.",
                 "What is the main request?",
                 [("refund", "the customer wants money returned"),
                  ("rebooking", "the customer wants a replacement flight"),
                  ("information", "the customer wants information only"),
                  ("other", "none of the above")])
    want = ("[kind] choice\n"
            "[state] I was charged twice for my flight to Berlin and nobody answers the phone.\n"
            "[question] What is the main request?\n"
            "[options]\n"
            "A. refund: the customer wants money returned\n"
            "B. rebooking: the customer wants a replacement flight\n"
            "C. information: the customer wants information only\n"
            "D. other: none of the above\n"
            "[decision]:")
    assert got == want, got
    assert render("noul", "s", "q", [("yes", ""), ("no", "")]).splitlines()[4] == "A. yes"
    assert len(LETTERS) == 16


def test_adapters_have_required_fields():
    assert len(adapters.ADAPTERS) >= 2
    for name, m in adapters.ADAPTERS.items():
        assert m.NAME == name, (m.NAME, name)
        assert m.NAME == Path(m.__file__).stem, m.__file__
        assert hasattr(m, "HF_ID")
        assert isinstance(m.KINDS, tuple) and m.KINDS and set(m.KINDS) <= set(KINDS), (name, m.KINDS)
        assert isinstance(m.LICENSE, str) and m.LICENSE
        assert isinstance(m.EVAL_ONLY, bool)
        assert callable(m.rows)
    assert adapters.get("jnli") is adapters.jnli
    try:
        adapters.get("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("an unregistered name should raise KeyError")


def test_configs_respect_data_rules():
    files = sorted((ROOT / "configs").glob("*.json"))
    assert files, "no json files under configs/"
    for f in files:
        cfg = cfgmod.load(f)
        check_sources(cfg)  # an EVAL_ONLY dataset among the training sources raises
        assert not set(cfg["train_sources"]) & set(cfg["eval_sources"]), f.name
        assert cfg["_root"] == ROOT, cfg["_root"]
        assert cfg["work_dir"] == ROOT / "runs" / Path(cfg["work_dir"]).name, cfg["work_dir"]
        assert cfg["data_dir"].parent == ROOT
        assert cfg["lora"]["r"] > 0 and cfg["train"]["max_len"] > 0

    # putting an evaluation-only dataset into the training sources must be rejected
    cfg = cfgmod.load(files[0])
    cfg["train_sources"] = ["livedoor"]
    try:
        check_sources(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("livedoor is ND-licensed and must not be allowed into training")


def test_cli_overrides_work_dir():
    cfg = cfgmod.load(ROOT / "configs" / "template.json", overrides={"work_dir": "runs/x_20260101"})
    assert cfg["work_dir"] == ROOT / "runs" / "x_20260101"


def test_make_row_and_shuffle():
    import random
    from japanese_jev.data_pipeline.common import make_row

    opts = [("a", "desc A"), ("b", ""), ("c", "desc C")]
    r = make_row("choice", "状況", "質問？", opts, answer_idx=2, source="t", rid=7, shuffle=False, rng=random.Random(0))
    assert r["label"] == "C" and r["completion"] == " C" and r["n"] == 3 and r["id"] == "7"
    assert r["prompt"] == render("choice", "状況", "質問？", opts)
    assert r["options"] == [["a", "desc A"], ["b", ""], ["c", "desc C"]]
    # after shuffling, label still points at option c
    for seed in range(20):
        s = make_row("choice", "状況", "質問？", opts, answer_idx=2, source="t", rid=7, shuffle=True,
                     rng=random.Random(seed), soft=[0.1, 0.2, 0.7])
        assert s["options"][LETTERS.index(s["label"])][0] == "c"
        assert abs(s["soft"][LETTERS.index(s["label"])] - 0.7) < 1e-6


def test_pick_parquet_files():
    from japanese_jev.data_pipeline.common import pick_parquet_files

    files = ["README.md", ".gitattributes",
             "base/train/0000.parquet", "base/test/0000.parquet",
             "original/train/0000.parquet", "stress/test/0000.parquet",
             "JNLI/train/0000.parquet", "JNLI/train/0001.parquet", "JNLI/validation/0000.parquet",
             "ja-JP/partial-train/0000.parquet"]
    assert pick_parquet_files(files, "base", "train") == ["base/train/0000.parquet"]
    assert pick_parquet_files(["x/train/0000.parquet"], None, "train") == ["x/train/0000.parquet"]   # a single config
    assert pick_parquet_files(["default/train/0.parquet", "b/train/0.parquet"], None, "train") == ["default/train/0.parquet"]
    assert pick_parquet_files(files, "JNLI", "train") == ["JNLI/train/0000.parquet", "JNLI/train/0001.parquet"]
    assert pick_parquet_files(files, "ja-JP", "train") == ["ja-JP/partial-train/0000.parquet"]
    for cfg, sp in (("nope", "train"), ("JNLI", "test"), (None, "train")):   # several configs and none requested -> error
        try:
            pick_parquet_files(files, cfg, sp)
        except RuntimeError:
            pass
        else:
            raise AssertionError((cfg, sp))


def test_build_dedupe_and_cap():
    import random
    from japanese_jev.data_pipeline.build import cap_per_kind, dedupe

    rows = [{"prompt": "p1", "label": "A", "kind": "choice", "source": "x"},
            {"prompt": "p1", "label": "A", "kind": "choice", "source": "x"},   # duplicate
            {"prompt": "p2", "label": "A", "kind": "noul", "source": "x"},
            {"prompt": "p2", "label": "B", "kind": "noul", "source": "y"}]     # conflict, the whole group is dropped
    kept = dedupe(rows)
    assert [r["prompt"] for r in kept] == ["p1"]

    pool = [{"prompt": f"c{i}", "label": "A", "kind": "choice", "source": "s1" if i % 2 else "s2"} for i in range(40)]
    pool += [{"prompt": f"n{i}", "label": "A", "kind": "noul", "source": "s3"} for i in range(5)]
    capped = cap_per_kind(pool, 10, random.Random(0))
    kinds = [r["kind"] for r in capped]
    assert kinds.count("choice") == 10 and kinds.count("noul") == 5
    srcs = {r["source"] for r in capped if r["kind"] == "choice"}
    assert srcs == {"s1", "s2"}, "both sources should be represented"


def test_augment_rerenders_and_keeps_label():
    import random
    from japanese_jev.data_pipeline.augment import augment_rows
    from japanese_jev.data_pipeline.common import make_row

    rows = [make_row("choice", f"前提: 文{i}　仮説: 文{i}", "関係は？", [("含意", "x"), ("矛盾", "y"), ("中立", "")],
                     answer_idx=i % 3, source="jnli", rid=i, shuffle=False, rng=random.Random(i)) for i in range(50)]
    extra = augment_rows(rows, 0.2, seed=1)
    assert 1 <= len(extra) <= 10
    for e in extra:
        assert e["source"].startswith("jnli+") and e["prompt"] != rows[0]["prompt"]
        assert e["prompt"] == render(e["kind"], e["state"], e["question"], [tuple(o) for o in e["options"]])
        orig = next(r for r in rows if e["id"].startswith(r["id"] + "+"))
        gold = orig["options"][LETTERS.index(orig["label"])][0]
        assert e["options"][LETTERS.index(e["label"])][0] == gold      # the name of the correct option is unchanged
        if not e["source"].endswith("+permute"):
            assert e["label"] == orig["label"] and e["options"] == orig["options"] and e["state"] != orig["state"]
        else:
            assert e["state"] == orig["state"] and e["options"] != orig["options"]
    assert augment_rows(rows, 0.0, seed=1) == []


def test_metrics_and_confidence():
    import numpy as np
    from japanese_jev import metrics
    from japanese_jev.readout import confidence, decide

    L = np.array([[2.0, 0.0, -np.inf, -np.inf],     # right
                  [0.0, 2.0, -np.inf, -np.inf],     # right
                  [0.0, 0.0, 1.0, -np.inf],         # wrong (the answer is 0), with lower confidence than the first two
                  [1.0, 1.0, 1.0, 1.0]])            # uniform
    y = np.array([0, 1, 0, 3])
    P = metrics.softmax(L)
    assert P.shape == L.shape and np.allclose(P.sum(1), 1) and P[0, 2] == 0
    assert abs(metrics.accuracy(P, y) - 0.5) < 1e-9
    assert 0 <= metrics.ece(P, y) <= 1 and metrics.brier(P, y) > 0
    # larger T -> less confident -> the NLL of a uniform row is unchanged while that of confident rows changes; argmax is unchanged
    assert np.array_equal(metrics.softmax(L, 3.0).argmax(1), P.argmax(1))
    assert metrics.nll(1.0, L[:2], y[:2]) < metrics.nll(5.0, L[:2], y[:2])
    # automation rate: sorted by confidence, the first two are both right; with 0 errors allowed only they pass
    assert abs(metrics.automation_rate(P, y, 0.0) - 0.5) < 1e-9
    assert metrics.automation_rate(P, y, 1.0) == 1.0
    s = metrics.summary(L, y)
    assert set(s) >= {"accuracy", "ece", "brier", "nll", "automation_2pct", "automation_5pct"}
    # confidence: the official definition, 0 for uniform, 1 for all mass on one option
    assert abs(confidence([0.25, 0.25, 0.25, 0.25])) < 1e-9
    assert abs(confidence([1.0, 0.0, 0.0]) - 1.0) < 1e-9
    p = decide([2.0, 0.0], T=1.0)
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[1]
    assert decide([2.0, 0.0], T=100.0)[0] < p[0]


def test_kuci_expand_and_permute():
    import random
    from japanese_jev.data_pipeline.adapters.kuci import expand_options
    from japanese_jev.data_pipeline.augment import permute_row
    from japanese_jev.data_pipeline.common import make_row

    opts = [("a", ""), ("b", ""), ("c", ""), ("d", "")]
    pool = ["b", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    for k in (4, 8):
        got = expand_options(opts, pool, k, random.Random(k))
        assert got[:4] == opts and len(got) == 4 + k, got
        assert len({n for n, _ in got}) == len(got), "options must not repeat"
        assert all(n != "b" for n, _ in got[4:])      # an option already present is not borrowed again
    # make_row shuffles after borrowing; label still points at the correct option
    r = make_row("choice", "s", "q", expand_options(opts, pool, 6, random.Random(1)), answer_idx=2,
                 source="kuci", rid=1, shuffle=True, rng=random.Random(2))
    assert r["n"] == 10 and r["options"][LETTERS.index(r["label"])][0] == "c"

    # permute: the order changes, the correct option does not, soft labels follow
    base = make_row("choice", "s", "q", [("x", ""), ("y", ""), ("z", "")], answer_idx=1, source="t", rid=5,
                    shuffle=False, rng=random.Random(0), soft=[0.1, 0.8, 0.1])
    p = permute_row(base, random.Random(3))
    assert p is not None and p["options"] != base["options"] and p["source"] == "t+permute"
    assert p["options"][LETTERS.index(p["label"])][0] == "y" and p["soft"][LETTERS.index(p["label"])] == 0.8
    assert "_order" not in p and p["prompt"] == render("choice", "s", "q", [tuple(o) for o in p["options"]])
    assert permute_row(dict(base, kind="noul", n=2), random.Random(0)) is None


def test_jsts_levels_and_temperature_lookup():
    from japanese_jev.calibrate import n_bucket, temperature_for
    from japanese_jev.data_pipeline.adapters import jsts
    from japanese_jev import metrics
    import numpy as np

    assert len(jsts.LEVELS) == 6 and jsts.KINDS == ("score",) and not jsts.EVAL_ONLY
    assert [n_bucket(n) for n in (2, 3, 5, 6, 12, 16)] == ["2", "3-5", "3-5", "6-16", "6-16", "6-16"]
    temps = {"choice": 1.2, "noul": 1.3, "score": 1.1, "by_n": {"choice": {"3-5": 1.25, "6-16": 1.0}}}
    assert temperature_for(temps, "choice", 4) == 1.25
    assert temperature_for(temps, "choice", 9) == 1.0
    assert temperature_for(temps, "choice", 2) == 1.2          # no bucket -> fall back to the primitive's T
    assert temperature_for(temps, "score", 5) == 1.1
    assert temperature_for({}, "noul", 2) == 1.0
    P = np.array([[0.1, 0.6, 0.3, 0.0], [0.7, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.7]])
    assert abs(metrics.within1(P, np.array([2, 2, 0])) - 1 / 3) < 1e-9   # argmax 1,0,3 vs 2,2,0 -> only the first row is within 1 level


def test_calib_sources_cannot_overlap():
    from japanese_jev.data_pipeline.build import check_sources

    base = {"train_sources": ["jnli"], "eval_sources": ["jcommonsenseqa"], "calib_sources": ["sib200"]}
    check_sources(base)
    for bad in ({**base, "calib_sources": ["jnli"]}, {**base, "calib_sources": ["jcommonsenseqa"]}):
        try:
            check_sources(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_ja_ambiguity_items():
    import collections
    from japanese_jev.data_pipeline.adapters import ja_ambiguity as m

    items = m.load_items()
    assert len(items) == 300 and len({it["id"] for it in items}) == 300
    for it in items:
        assert it["kind"] in ("choice", "noul", "score") and 0 <= it["answer"] < len(it["options"]) <= 16
        assert it["phenomenon"] and it["scene"] and "\n" not in it["state"]
    rows = list(m.rows("test", seed=42))
    assert len(rows) == 300 and m.EVAL_ONLY
    letters = collections.Counter(r["label"] for r in rows if r["kind"] == "choice")
    assert max(letters.values()) < 0.5 * sum(letters.values()), "the correct option of choice rows should not cluster on one letter after shuffling"
    assert all(r["label"] in "AB" for r in rows if r["kind"] == "noul")


def test_ja_knowledge_items():
    import collections
    from japanese_jev.data_pipeline.adapters import ja_knowledge as m

    items = m.load_items()
    assert len(items) == 200 and len({it["id"] for it in items}) == 200 and m.EVAL_ONLY
    for it in items:
        assert it["kind"] in ("choice", "noul") and 0 <= it["answer"] < len(it["options"]) <= 16
    rows = list(m.rows("test", seed=42))
    letters = collections.Counter(r["label"] for r in rows if r["kind"] == "choice")
    assert max(letters.values()) < 0.5 * sum(letters.values())


def test_ja_negation_balanced():
    import collections
    from japanese_jev.data_pipeline.adapters import ja_negation as m

    items = m.load_items()
    assert len(items) == 100 and collections.Counter(it["answer"] for it in items) == {0: 50, 1: 50}
    assert collections.Counter(it["phenomenon"] for it in items) == {"肯定": 25, "単純否定": 25, "二重否定": 25, "否定の結論": 25}


def test_split_by_group_keeps_same_state_together():
    import random
    from japanese_jev.data_pipeline.build import split_by_group

    pool = [{"source": "jsick", "state": f"s{i}", "prompt": f"s{i}-{k}", "kind": k} for i in range(200) for k in ("noul", "score")]
    valid, calib, train = split_by_group(pool, 50, 50, random.Random(0))
    sv, sc, st = ({r["state"] for r in part} for part in (valid, calib, train))
    assert not (sv & st) and not (sc & st) and not (sv & sc)
    assert len(valid) >= 50 and len(calib) >= 50 and len(valid) + len(calib) + len(train) == 400


def test_livedoor_mask_and_stratify():
    import collections, random
    from japanese_jev.data_pipeline.adapters.livedoor import mask_site, stratified

    assert "Sports Watch" not in mask_site("【Sports Watch】石井慧が電撃離婚か？") and mask_site("【Sports Watch】試合").startswith("試合")
    assert "Peachy" not in mask_site("Peachyの記事、独女通信より")
    recs = [{"category": c, "i": i} for c in range(9) for i in range(30 if c else 300)]
    top = stratified(recs, lambda r: r["category"], random.Random(0))[:90]
    assert set(collections.Counter(r["category"] for r in top).values()) == {10}


def test_serve_options_and_answer_shape():
    from japanese_jev.serve import build_options

    assert build_options("noul", {}) == [("yes", ""), ("no", "")]
    assert build_options("score", {"criteria": ["低", "中", "高"]}) == [("0", "低"), ("1", "中"), ("2", "高")]
    assert build_options("choice", {"criteria": {"返金": "お金", "予約変更": None}}) == [("返金", "お金"), ("予約変更", "")]
    for bad in ({"criteria": "x"}, {"criteria": ["only"]}):
        try:
            build_options("choice", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all tests passed")
