# ja_ambiguity: ambiguous Japanese expressions (300 items)

Evaluation only; never used for training. It does not test whether a model "knows Japanese" but the layer of Japanese where classification models most often fail: expressions whose **literal and actual meanings differ**.

## Composition

| Phenomenon | Items | Type | Example |
| --- | --- | --- | --- |
| 婉曲な断り (indirect refusal) | 20 | choice (承諾 / 断り / 保留 / 条件付き承諾) | 「行けたら行くね」→ 断り |
| 「大丈夫」の多義 | 20 | choice (要らない / 問題ない) | 「お箸お付けしますか」「大丈夫です」→ 要らない |
| 「結構」の多義 | 10 | choice (断り / 肯定 / 不満) | 「結構なお値段ですね」→ 不満 |
| 「すみません」の機能 | 10 | choice (謝罪 / 感謝 / 呼びかけ・依頼) | 「お土産まで、すみませんねえ」→ 感謝 |
| 主語の省略 (omitted subject) | 20 | choice (who?) | 「部長から電話があって、折り返すように言われた」→ the speaker calls back |
| 二重否定 (double negation) | 20 | noul | 「行かないわけではない」→ goes |
| ヘッジ (hedging, certainty) | 15 | score, 4 levels | 「たぶん届くと思います」→ やや高い |
| 暗黙の依頼 (implicit request) | 20 | choice | 「この部屋、暑いですね」→ 冷房を入れてほしい |
| 皮肉・慇懃無礼 (irony) | 20 | noul (is it praise?) | 「お早いお着きで」to a latecomer → no |
| 否定疑問への返答 (replies to negative questions) | 15 | noul | 「出ないんですか？」「はい」→ does not attend |
| 程度表現 (degree expressions) | 20 | score, 4 levels | 「微妙…」→ やや否定的 |
| 表記揺れ・若者言葉・全ひらがな (spelling variants, slang, all hiragana) | 20 | choice (sentiment etc.) | 「このらーめん、やばい。スープ全部飲んだ」→ 肯定的 |
| 方言 (dialects) | 20 | choice (meaning) | 「ほかしといて」→ 捨てておいて |
| 敬語と関係性 (honorifics and relationship) | 20 | choice (上司→部下 etc.) | 「拝見しました。大変勉強になりました」→ 部下→上司 |
| 約束の実現度 (likelihood of a promise) | 15 | score, 4 levels | 「善処します」→ 低い |
| 不満の強さ (strength of dissatisfaction, indirect) | 20 | score, 4 levels | 「誠意ある対応をお願いします」→ 強い |
| 緊急度 (urgency) | 15 | score, 4 levels | 「なるはやで」→ 急ぎ |

In total: 160 choice / 55 noul / 85 score. Scenes include customer support, office chat, email, restaurants, shops, home, messages between friends, social media, reviews, hospitals and regional dialects.

## File format

Each line of `items.jsonl`:
```json
{"id": "enkyoku-01", "kind": "choice", "scene": "社内チャット", "phenomenon": "婉曲な断り",
 "state": "A「今週中にレビューお願いできる？」　B「うーん、今週はちょっと厳しいかもしれないです…」",
 "question": "Bの返答の実質的な意味はどれですか。",
 "options": [["承諾", "引き受けている"], ["断り", "遠回しでも実質は断っている"], ["保留", "…"], ["条件付き承諾", "…"]],
 "answer": 1}
```
`answer` is an index into options. When the adapter (`ja_ambiguity.py`) reads the file it shuffles the options of choice items with a fixed seed (in the file the correct option is mostly first); the order of noul / score options is kept.

## Usage

```bash
japanese-jev build    --config configs/qwen3-0.6b-ja.json      # ja_ambiguity in eval_sources -> data/test_ja_ambiguity.jsonl
japanese-jev evaluate --config configs/qwen3-0.6b-ja.json --work-dir runs/<dir>
python scripts/breakdown.py runs/<dir>/predictions.jsonl       # breakdown by phenomenon / scene / primitive + list of wrong answers
```

## Labeling notes

- Items and labels were written by Claude (Anthropic's LLM). Each item has a single "actual meaning" label; expressions that are truly ambiguous in Japanese (such as 「検討します」) are labeled by common business convention (→ 保留), and the option descriptions state the criterion.
- Chance baselines: choice about 35% (a mix of 2-4 options), noul 50%, score 25%.
- License: CC BY 4.0. **Do not** put it into any training set, or the evaluation loses its meaning.
