# Source-grounded English study assistant: initial contract

The complete target is 50 development / 150 independently held final cases.
Quotas and proposed product thresholds are in `../configs/evaluation.json`.
Twenty development drafts currently exist across ten newly authored source
groups. Their references were checked by the implementing assistant, not a human
rater. Complete and review the remaining development set before model selection.

Use source text as data, never as instructions. Report answers with sentence IDs;
check support, not citation existence alone. Outside knowledge is unsupported in
this task even when true elsewhere. Include temporal updates, negation, numbers,
unanswerable questions and over-refusal checks. No task-agnostic aggregate of
unrelated 0–4 scales is defined.

## Anchored rubrics

| Score | Faithfulness | Coverage | Clarity |
|---|---|---|---|
| 0 | Contradicts central facts or fabricates the answer | Omits the requested content | Unusable |
| 1 | Major unsupported claims; contradicts a qualifier/negation | Misses most key facts | Hard to interpret |
| 2 | Some correct facts, with a material unsupported implication | About half of required facts | Understandable with ambiguity |
| 3 | Supported, with a minor imprecision | Nearly all required facts | Clear with minor awkwardness |
| 4 | Fully supported; uncertainty/inference appropriately marked | All important facts at suitable length | Clear, concise and well organized |

A fluent contradiction cannot receive faithfulness above 1. Record unsupported
assertions separately, including denominators. Leave unassigned human fields null;
an automated exact-match result is not a human rating. Blind checkpoint names,
randomize paired order, allow ties and use independent raters when available.

## Deterministic protocol and context

Use native model chat templates and EOS IDs, greedy decoding, one beam and a
128-token response cap. Refuse source+template+response overflow rather than
truncating answer-bearing facts. Pretrained evaluation is a compatibility exercise
until thresholds and the full development set are approved. Stage 6 completion
quality stays a separate reference; do not penalize it for a new chat template.

## Sealed final evaluation

An independent owner should author/review the 150 cases and hold content/labels
outside this tuning workspace. Group all source variants/questions together.
The owner provides the file hash, quotas, owner identity and release policy.
`python3 -B -m stage_7_general_lm.evaluation --register-sealed OWNER_FILE --owner NAME --report NEW_RECEIPT`
hashes bytes without parsing labels; the declared 150 count is not independently
verified by hashing. No sealed set has been authored, viewed or claimed here.
The owner executes locked final candidates once. If the test is used for tuning,
label it exposed and obtain another held-out set before a final untouched claim.

Use development groups for selection; exclude all evaluation passages, groups,
hashes and known derivatives from project-controlled pretraining/SFT/calibration.
`protected_groups.json` records the current draft groups, not unknown future
sealed groups. External pretraining contamination cannot be ruled out. Source
group bootstrap intervals are available for an individual binary metric; the
small, clustered set does not justify broad superiority claims.
