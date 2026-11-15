# ja_knowledge: knowledge of Japan and the Japanese language (200 items)

Where `ja_ambiguity` tests pragmatics (literal vs actual meaning), this set tests knowledge: things a person living and working in Japan is assumed to know. Evaluation only; never used for training.

| Category | Items | Type | Example |
| --- | --- | --- | --- |
| 慣用句・ことわざ (idioms, proverbs) | 20 | choice, 4-way | 「油を売る」→ 無駄話をして怠ける |
| 四字熟語 (four-character idioms) | 10 | choice, 4-way | 「本末転倒」 |
| 敬語の正誤 (correct honorifics) | 20 | noul | 「拝見させていただきます」→ double honorific, incorrect |
| 漢字の読み (kanji readings) | 15 | choice, 4-way | 「相殺」→ そうさい |
| 和製英語・カタカナ語 (loanwords) | 15 | choice, 4-way | 「コンセント」→ 電源の差し込み口 |
| 助数詞 (counters) | 10 | choice, 4-way | うさぎ → 羽 |
| 生活・社会制度 (daily life, institutions) | 25 | choice, 4-way | standard consumption tax rate 10%, police 110 |
| ビジネスマナー (business etiquette) | 10 + 10 | choice, 4-way / noul | hand over a business card with both hands; keep 「Re:」 |
| 地理 (geography) | 15 | choice, 4-way | capital of 兵庫県 → 神戸市 |
| 歴史・年号 (history, eras) | 15 | choice, 4-way | 昭和 1926-1989 |
| 文化・年中行事 (culture, annual events) | 20 | choice, 4-way | 節分 → 豆まき |
| 擬音語・擬態語 (onomatopoeia) | 15 | choice, 4-way | 「ぺこぺこ」→ 空腹 |

In total: 170 choice / 30 noul. Chance baselines: choice 25%, noul 50%.

Format and usage are the same as in `../ja_ambiguity/README.md` (the `scene` field holds the subcategory here). Breakdown by category:

```bash
python scripts/breakdown.py runs/<dir>/predictions.jsonl evalsets/ja_knowledge/items.jsonl
```

Labeling notes: all answers are stable facts (nothing time-sensitive such as current office holders). Items and labels were written by Claude (Anthropic's LLM). Honorifics are judged by common guidelines (double honorifics, humble forms misapplied to the listener and "baito keigo" count as incorrect). Released under CC BY 4.0; **do not put it into a training set**.

## Versions

- **v1.1**: the question wording of 45 items is stricter than in v1.0 (texts, options and labels are unchanged). Some wrong options are also in common use today (うさぎ → 匹, 続柄 → ぞくがら), which makes an item ambiguous under a question like 「正しいものはどれか」. The wording was changed to:
  - 助数詞, 10 items: 「これを数えるとき、伝統的に正式とされる助数詞はどれですか。」
  - 漢字の読み, 15 items: 「この語の本来の読み方（辞書で最初に挙げられる読み）はどれですか。」
  - 敬語の正誤, 20 items: 「この表現は、ビジネスマナーの本で正しいとされる敬語ですか。」
- v1.0: the original 200 items. **Scores on v1.0 and v1.1 are not directly comparable.**
