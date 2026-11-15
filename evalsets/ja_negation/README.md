# ja_negation: negation minimal pairs (100 items)

Built to check whether training taught the model to "answer no whenever a negation word appears". 25 groups; in each group the same question is paired with 4 sentences:

| Type | Example | Answer |
| --- | --- | --- |
| 肯定 (affirmative) | 明日の会には行きます。 | yes |
| 単純否定 (plain negation) | 明日の会には行きません。 | no |
| 二重否定 (double negation) | 明日の会に行かないわけではありません。 | yes |
| 否定の結論 (negative conclusion) | 明日の会には行かないことにしました。 | no |

50 yes and 50 no. How to read the breakdown (`python scripts/breakdown.py runs/<dir>/predictions.jsonl evalsets/ja_negation/items.jsonl`):
- all four types high: the model handles negation.
- affirmative, plain negation and negative conclusion high, **double negation low**: a negation-word shortcut (about 75% overall).
- always yes or always no: 50% overall.

The 20 double-negation items in `ja_ambiguity` are skewed (16 of them are yes) and can only hint at a problem; use this set to draw conclusions.
Items and labels were written by Claude (Anthropic's LLM). Evaluation only; never used for training.
