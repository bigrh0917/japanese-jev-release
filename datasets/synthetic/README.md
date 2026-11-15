# Synthetic data

The synthetic data used for training. Texts and labels were written by Claude (Anthropic's AI). CC BY 4.0.

| File | Items | Contents | Generator | Used in the final models |
| --- | --- | --- | --- | --- |
| `train.jsonl` | 1,109 | negation minimal pairs, litotes such as 「悪くない」, replies to negative questions, indirect replies, replies to offers, honorifics and hierarchy, likelihood of a promise being kept, certainty, urgency, strength of dissatisfaction, emotion intensity, idioms | `scripts/make_synthetic_v1.py` | yes |
| `train_v2.jsonl` | 1,965 | 280 short texts in 9 scenes (business email, office chat, customer inquiries, reviews, social media, notices, messages between friends, shop conversations, school and neighborhood notices), with questions about intent, sentiment, topic, urgency, politeness, strength of dissatisfaction, whether a reply is needed, whether there is a deadline | `scripts/make_synthetic_v2.py` | no (adding it made no measurable difference) |

One item per line.

```json
{"id": "neg-00001", "kind": "noul", "state": "その映画を見ません。", "question": "話し手はその映画を見ますか。", "options": [["yes", ""], ["no", ""]], "answer": 1}
```

`options` is a list of `[name, description]` and `answer` is the position of the correct option (0-based). score options are ordered from the lowest level to the highest.

Notes:

- The data has the same author (Claude) as the evaluation sets in `evalsets/`. Texts with a large character 3-gram overlap were removed so that no text is shared, but the writing habits are the same.
- It has not been reviewed by a native speaker of Japanese.
