"""Adapter registry: one module per dataset, referenced from cfg by NAME.

Every adapter module must define (tests/test_pipeline.py checks this):
    NAME       the name used in cfg, same as the file name
    HF_ID      HuggingFace dataset id; None if it is not on HF, with the download URL in a comment (e.g. mewsc16)
    KINDS      the primitives it can produce, a tuple such as ("choice",) / ("noul", "score")
    LICENSE    the license
    EVAL_ONLY  True = the license does not allow training (ND, research-only, Amazon terms); eval_sources only
    rows(split)  generator yielding rows in the unified format; split is "train" / "valid" / "test"

Training:           jnli  jsick  jsts  kuci  pawsx_ja  mewsc16  synthetic  jcola_train (continued training only)
Temperature only:   sib200 (calib_sources in the config; neither trained on nor evaluated)
Evaluation:         jcommonsenseqa  massive_ja  livedoor  wrime  jcola  ja_ambiguity (self-built, 300 items, pragmatics)
                    ja_knowledge (self-built, 200 items, knowledge)  ja_negation (self-built, 100 items, negation)
"""

from __future__ import annotations

from types import ModuleType

from . import (
    ja_ambiguity,
    ja_knowledge,
    ja_negation,
    jcola,
    jcola_train,
    jcommonsenseqa,
    jnli,
    jsick,
    jsts,
    kuci,
    livedoor,
    marc_ja,
    massive_ja,
    mewsc16,
    pawsx_ja,
    sib200,
    synthetic,
    wrime,
)

ADAPTERS: dict[str, ModuleType] = {
    m.NAME: m
    for m in (jnli, jsick, jsts, kuci, pawsx_ja, mewsc16, jcola_train, sib200, synthetic,
              jcommonsenseqa, massive_ja, livedoor, wrime, marc_ja, jcola, ja_ambiguity, ja_knowledge, ja_negation)
}


def get(name: str) -> ModuleType:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"no adapter named {name!r}; available: {sorted(ADAPTERS)}") from None
