"""Japanese augmentation: training set only, labels unchanged, 10-20% of the training set.

  hiragana   all hiragana                      pykakasi (skipped if not installed)
  zenkaku    half-width -> full-width          jaconv (skipped if not installed)
  hankaku    full-width -> half-width (digits, Latin letters, katakana)   jaconv
  noise      append an unrelated sentence to state   no dependencies
  permute    reorder the options of a choice item    only for choice with n >= 3; the order of noul / score options is meaningful and kept

The first four rewrite the state text and permute changes the option order; all of them call template.render()
again rather than editing the prompt string.
Honorific / colloquial rewriting would need a generative model and is not done here.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..template import LETTERS, render

# sentences unrelated to the item that change no decision; appended to state
NOISE_SENTENCES = [
    "なお、本日は晴れです。",
    "ちなみに、昨日は雨でした。",
    "ところで、今日は火曜日です。",
    "念のため、この文は無関係です。",
    "ご参考までに、会議は午後三時からです。",
    "追記：特に変更はありません。",
    "ちなみに、コーヒーを一杯飲みました。",
    "備考：この行は読み飛ばして構いません。",
]


def _hiragana_fn() -> Callable[[str], str] | None:
    try:
        import pykakasi
    except ImportError:
        return None
    kks = pykakasi.kakasi()
    return lambda s: "".join(item["hira"] for item in kks.convert(s))


def _zenkaku_fn() -> Callable[[str], str] | None:
    try:
        import jaconv
    except ImportError:
        return None
    return lambda s: jaconv.h2z(s, kana=True, digit=True, ascii=True)


def _hankaku_fn() -> Callable[[str], str] | None:
    try:
        import jaconv
    except ImportError:
        return None
    return lambda s: jaconv.z2h(s, kana=True, digit=True, ascii=True)


def _noise_fn(rng: random.Random) -> Callable[[str], str]:
    return lambda s: s + "　" + rng.choice(NOISE_SENTENCES)


def transforms(rng: random.Random) -> dict[str, Callable[[str], str]]:
    """name -> function that rewrites state. Which transforms exist depends on the installed packages; noise is always available. permute is not here because it changes the options."""
    out: dict[str, Callable[[str], str]] = {}
    for name, fn in (("hiragana", _hiragana_fn()), ("zenkaku", _zenkaku_fn()), ("hankaku", _hankaku_fn())):
        if fn is not None:
            out[name] = fn
        else:
            print(f"  [augment] {'pykakasi' if name == 'hiragana' else 'jaconv'} is not installed, skipping {name}")
    out["noise"] = _noise_fn(rng)
    return out


def _finish(row: dict, name: str, state: str, options: list[tuple[str, str]], answer_idx: int) -> dict:
    new = dict(row)
    new["state"] = state
    new["options"] = [[n, d] for n, d in options]
    new["n"] = len(options)
    new["label"] = LETTERS[answer_idx]
    new["completion"] = " " + LETTERS[answer_idx]
    new["prompt"] = render(row["kind"], state, row["question"], options)
    new["source"] = f"{row['source']}+{name}"
    new["id"] = f"{row['id']}+{name}"
    order = new.pop("_order", None)            # the new order passed in by permute; soft labels are reordered with it
    if "soft" in row and order is not None:
        new["soft"] = [row["soft"][i] for i in order]
    return new


def augment_row(row: dict, name: str, fn: Callable[[str], str]) -> dict | None:
    """Apply one state transform to a row and return the new row; None if state did not change. The label is unchanged."""
    state = fn(row["state"])
    if not state or state == row["state"]:
        return None
    options = [tuple(o) for o in row["options"]]
    return _finish(row, name, state, options, LETTERS.index(row["label"]))


def permute_row(row: dict, rng: random.Random) -> dict | None:
    """Reorder the options of a choice item; the label follows the correct option. None if the order did not change (or there are fewer than 3 options)."""
    if row["kind"] != "choice" or row["n"] < 3:
        return None
    options = [tuple(o) for o in row["options"]]
    order = list(range(len(options)))
    for _ in range(5):
        rng.shuffle(order)
        if order != list(range(len(options))):
            break
    else:
        return None
    gold = LETTERS.index(row["label"])
    new = dict(row)
    new["_order"] = order
    return _finish(new, "permute", row["state"], [options[i] for i in order], order.index(gold))


def augment_rows(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    """Sample a ratio of rows, apply one random transform to each, and return the new rows (without the originals)."""
    if ratio <= 0 or not rows:
        return []
    rng = random.Random(f"augment-{seed}")
    fns = transforms(rng)
    names = sorted(fns) + ["permute"]
    n = int(len(rows) * ratio)
    picked = rng.sample(rows, min(n, len(rows)))
    out = []
    for row in picked:
        if "state" not in row or "options" not in row:
            continue                       # rows without the raw fields (state / question / options) are skipped
        name = rng.choice(names)
        if name == "permute":
            new = permute_row(row, rng)
            if new is None:                # not a choice item: fall back to a state transform
                name = rng.choice(sorted(fns))
                new = augment_row(row, name, fns[name])
        else:
            new = augment_row(row, name, fns[name])
        if new is not None:
            out.append(new)
    return out
