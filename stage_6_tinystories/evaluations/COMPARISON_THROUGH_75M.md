# Stage 6A TinyStories: rigorous +25M / +50M / +75M comparison

Audit date: **2 November 2026**, Asia/Kolkata. Analysis of existing outputs only; no training, generation or model evaluation was run. Existing source, checkpoints, evaluation artifacts, ratings and reports are preserved.

## 1. Executive summary

**+75M improves recorded next-token prediction, but does not establish a broad, meaningful storytelling improvement over +50M.** Development CE falls from **2.094938 to 2.065159** (−0.029779 nats/token; **1.4215%**), and development perplexity from **8.124939 to 7.886555** (**2.9340%**). Train CE improves almost equally. This is continued language-modelling progress under the recorded evaluation protocol.

The generated stories show **local gains, persistent weaknesses and direct regressions**. The +75M zoo story has a cleaner visit–play–home arc, and one fox/treasure story retains the fox better. Conversely, Rose becomes Emma, Eliza again drifts into several unrelated names, and one book-reading continuation becomes more repetitive. All three checkpoints still change Tom's toy location from “under the bed” to “under a bush” in the same manual-probe seed.

Automated diagnostics are mixed. EOS improves from **151/159 to 154/159**, returning to the +25M rate. Mean continuation length falls **13.29%**, from 155.107 to 134.497 tokens. Mean repeated-token-4-gram fraction falls only slightly, from 0.032356 to 0.031453, while the number of samples at or above 0.10 rises from **5 to 8**. The worst +50M loop shrinks, but semantic contradictions and empty endings remain.

**No human quality scores are available:** all 1,908 rating cells across the three CSVs are blank. The qualitative conclusions below are my unblinded, model-assisted close reading, not external human ratings. All **477** saved outputs were checked programmatically; **60 matched triplets / 180 continuations** were read in detail, covering every development prompt and all manual-probe seeds. This supports a mixed assessment, not a numerical population-wide preference estimate.

There is a sign of **smaller marginal CE gains**: the latest equal-sized training interval improves dev CE by about **23.2% less** than the preceding interval. There is **no obvious aggregate development-loss overfitting signal**: dev CE keeps falling and the train–dev gap barely changes. Neither a plateau nor memorization can be established from three checkpoints.

**Repository consolidation note:** the verified complete +25M set now has the
canonical path `evaluations/25m/`. The duplicate complete set and incomplete
generation attempt were removed from the active tree during the final cleanup.

## 2. Sources, comparability and validation

### Actual input sets

| Milestone | Actual complete evaluation directory | Check |
|---|---|---|
| +25M | [evaluations/25m](25m/evaluation.json) | Complete CE plus 159 generations; update 3052 |
| +50M | [evaluations/50m](50m/evaluation.json) | Complete CE plus 159 generations; update 6104 |
| +75M | [evaluations/75m](75m/evaluation.json) | Complete CE plus 159 generations; update 9156 |

All three scores and stories in this report come from their respective **complete**
evaluation records. A byte-identical +25M copy and an incomplete +25M attempt were
excluded and removed from the active tree.

Recorded evaluation checkpoint identities:

| Milestone | Recorded checkpoint basename | Recorded SHA256 |
|---|---|---|
| +25M | `best.pt` | `d70946e39078e9474cae34742c8ad7d96d0ba6315a2b849b75aed12dbd742315` |
| +50M | `latest.pt` | `40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e` |
| +75M | `review_u9156_additional_75005952.pt` | `7bcad3a13ce5e056b11af0617e6890db072a7b4f5eaafb5df9fdc8b47f5e9105` |

The local +25M `best.pt` was rehashed and matches its evaluation record. +50M and +75M hashes above were read from evaluation JSONs, not freshly compared with remote checkpoint bytes. A rolling `latest.pt` path is a locator, not a durable milestone identity; the recorded SHA matters. This report is a saved-output comparison, not a new audit of +75M training or receipt/PBS provenance.

### Equal recorded evaluation protocol

All three records agree on:

- Data manifest `89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7`; tokenizer `99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038`; prompt set `f1c83474faa2b56d670055d6da373a1460eccc007db296bcf509f44f5737429c`.
- Evaluation seed 2026, context 256, batch 16; train 512 windows / **131,072 scored positions**; dev_val 4096 windows / **1,048,576 scored positions**.
- Generation: **53 prompts** (50 development prefixes and 3 manual probes), each at seeds **1337, 2027, 3407**; temperature **0.7**, top-k **50**, maximum **300 new tokens**.
- Recorded runtime: V100-PCIE-32GB, Python 3.7.9, PyTorch 1.10.0a0+git36449ea, NumPy 1.18.5, CUDA 11.0, FP32/manual causal attention and identical CUBLAS setting.

The JSONs record equal settings/data/runtime, but do not attest execution-time evaluator/generator source hashes or save every scored window. Matching records are therefore strong comparability evidence, not proof of identical executed code or a bitwise replay. The historical generation-only model patch is not assumed to have run identically at all three gates. No model was rerun to close this uncertainty.

### Checks performed on saved artifacts

For every one of the 477 stories:

1. Exactly one row exists for each expected `(prompt_id, seed)`; no missing, duplicate or additional key. Prompt text and kind match the saved prompt bank and match across milestones.
2. Saved generated IDs are valid nonspecial vocabulary IDs, decode to the exact stored continuation, and have the reported `new_tokens` count.
3. Repeated-token-4-gram fraction recomputes from saved IDs. EOS counts, mean length and mean repetition reproduce the evaluation JSON totals.
4. CSV prompt/story/stop/length/repetition values match the corresponding JSONL row.
5. Each complete set has 159/159 rows, `generation_complete=true` and `status=complete`; stop reasons are only EOS or the max-token cap.
6. Stored perplexity equals `exp(CE)`. CE itself is read from saved records; it was not recomputed by running a model.
7. Each CSV has 159 rows and four rating columns, all blank: grammar, character consistency, prompt facts and ending. There are **0/636** filled rating cells per milestone.

Processed positions below are training exposure, not necessarily unique corpus tokens. Continuation lengths exclude prompt tokens and the terminating EOS token.

## 3. Quantitative comparison

### Milestone table

| Quantity | +25M | +50M | +75M |
|---|---:|---:|---:|
| Stage 6 update | 3,052 | 6,104 | 9,156 |
| Additional processed positions | 25,001,984 | 50,003,968 | 75,005,952 |
| Total processed positions | 123,305,984 | 148,307,968 | 173,309,952 |
| Evaluation-mode train CE (nats/token) | 2.083491 | 2.044402 | 2.014335 |
| Development CE (nats/token) | 2.133726 | 2.094938 | 2.065159 |
| Train perplexity | 8.032458 | 7.724536 | 7.495740 |
| Development perplexity | 8.446280 | 8.124939 | 7.886555 |
| Dev CE minus train CE | 0.050235452 | 0.050536491 | 0.050824592 |
| EOS fraction | 154/159 (96.855%) | 151/159 (94.969%) | 154/159 (96.855%) |
| Max-token terminations | 5 | 8 | 5 |
| Mean continuation length (tokens; EOS excluded) | 148.931 | 155.107 | 134.497 |
| Median continuation length (tokens) | 148 | 147 | 132 |
| Mean repeated-token-4-gram fraction | 0.034323 | 0.032356 | 0.031453 |
| Median repeated-token-4-gram fraction | 0.026846 | 0.021583 | 0.021127 |
| Maximum repeated-token-4-gram fraction | 0.178261 | 0.331754 | 0.152381 |
| Samples with repeated-4-gram fraction >=0.10 | 8/159 | 5/159 | 8/159 |
| Samples with repeated-4-gram fraction >=0.20 | 0/159 | 1/159 | 0/159 |
| Samples with zero repeated 4-grams | 21/159 | 18/159 | 29/159 |
| Generated / expected samples | 159/159 | 159/159 | 159/159 |
| Status / generation complete | complete / True | complete / True | complete / True |
| Human-rating fields filled | 0/636 | 0/636 | 0/636 |

The train scores are evaluation-mode scores on fixed training-split windows, not dropout-enabled optimization minibatch losses. Perplexity is a transformation of CE, so its improvement is not an independent second measurement.

### Prediction trends

Signed CE changes below use later minus earlier; negative is better.

| Interval | Train CE change | Dev CE change | Dev CE reduction | Dev perplexity reduction |
|---|---:|---:|---:|---:|
| 25→50M | -0.039088946 | -0.038787906 | 1.8178% | 3.8045% |
| 50→75M | -0.030066889 | -0.029778788 | 1.4215% | 2.9340% |
| 25→75M | -0.069155835 | -0.068566694 | 3.2135% | 6.6269% |

The +50M→+75M train improvement is **0.030067**, close to the dev improvement **0.029779**. The generalization gap rises only from **0.050536491 to 0.050824592**: +0.000288100 nats/token. Across +25M→+75M it rises by about 0.000589140. This does not look like train loss improving while development loss deteriorates.

The equal-sized intervals each add 25,001,984 processed positions. Their dev CE improvements are 0.038787906 and 0.029778788: the second is **23.23% smaller**. That is reduced marginal improvement at the endpoints, not evidence that the loss has stopped improving. With only three gates, the within-interval curve and the contribution of the original LR decay cannot be separated.

### Termination, length and repetition are different questions

- **EOS:** +75M recovers the +25M 154/159 result, rather than setting a new high. Against +50M, five previously capped pairs now terminate with EOS, while two previously EOS pairs now cap: `dev_10452/2027` and `dev_10452/3407`. This is improvement in some cases and regression in others, not monotonic repair.
- **Length:** +75M is **20.610 tokens shorter on average** than +50M (−13.29%), and 14.434 shorter than +25M. Median length falls from 147 to 132 versus +50M. Shorter text can remove detours; it can also end before a meaningful problem is resolved.
- **Exact token repetition:** the lower +75M average partly coexists with a repaired extreme +50M outlier. Its maximum falls from 0.3318 to 0.1524; however, samples ≥0.10 increase from 5 to 8. The 0.10 threshold is a descriptive tail check, not a validated quality cutoff.
- **All eight** +75M samples above that repetition threshold end with EOS. EOS is plainly not a filter for loops.
- **Hard prompts remain hard:** all three +75M `dev_06045` seeds cap at 300; two `dev_10452` seeds cap. The net cap count of five hides that concentration.
- **No literal text is unchanged:** no complete continuation is byte-identical between any two milestones for a matched key. “Unchanged” in the qualitative assessment means similar capability/weakness, not identical output.

Here, repeated-4-gram fraction is `1 − unique_4grams / total_4grams` over consecutive **token-ID** 4-grams, averaged equally across stories. It is not word-level redundancy or semantic repetition. It misses paraphrased loops, entity swaps and unsupported morals; output length also changes the opportunity to repeat.

### Exploratory paired uncertainty for generation diagnostics

A prompt-cluster bootstrap resampled **53 prompt IDs**, retaining all three seeds within each selected prompt; 20,000 resamples, private analysis RNG seed 20261102. This avoids treating correlated seeds for the same prompt as 159 independent prompt observations. It only postprocesses saved rows; no model/evaluation RNG state was loaded or restored.

Intervals are exploratory 95% percentile intervals over this fixed prompt bank, not guarantees for other prompts, languages, training runs or human preference. No multiplicity-adjusted hypothesis tests are claimed.

| Interval | Mean length change, tokens [95% interval] | EOS change, percentage points [95% interval] | Mean repeated-4-gram change [95% interval] |
|---|---:|---:|---:|
| +25M to +50M | 6.176 [-2.126, 14.472] | -1.887 [-4.403, 0.629] | -0.001967 [-0.008718, 0.004996] |
| +25M to +75M | -14.434 [-21.931, -6.912] | 0.000 [-1.887, 1.887] | -0.002869 [-0.008632, 0.002981] |
| +50M to +75M | -20.610 [-28.528, -12.648] | 1.887 [-0.629, 4.403] | -0.000902 [-0.007119, 0.005475] |

The +50M→+75M **length reduction is consistent** under this resampling. Intervals for EOS and mean repetition include zero; their small average changes are weak evidence for an underlying generation-distribution improvement. None of these intervals measures story quality.

CE uncertainty intervals cannot be responsibly computed from scalar aggregate losses alone. Saved per-window paired losses are unavailable, and overlapping token windows are not independent observations.

## 4. Qualitative method and dimension-by-dimension assessment

### Review scope and limitations

The systematic set sorts the **53 prompt IDs**, assigning seeds cyclically 1337/2027/3407 by sorted index. This gives one matched triplet for every prompt. All remaining manual-probe seeds were then included: **59 matched triplets**. One extra triplet, `dev_01952/2027`, was selected explicitly because it had +50M's worst repetition: **60 triplets / 180 full continuations read** in total.

This covers every one of the 50 development-prefix prompts and all nine manual prompt/seed combinations. The reviewed seed counts are 20/21/19. It does **not** mean every one of the 159 triplets was manually read. The 477-output programmatic checks and 180-continuation close reading are separate.

Milestones were visible during reading; this is not blinded or preregistered human assessment, has no inter-rater reliability estimate and does not produce invented 0–4 scores, quality averages or a preference win rate. Appendix B lists the reviewed keys and brief observations. The examples below illustrate both directions and mixed cases; they are not frequency estimates.

Talking animals, magical objects and unusual events are allowed within TinyStories-style fantasy. The concerns are unstable identities, contradictory states, missing causal bridges and unearned endings—not fantasy by itself.

| Dimension | Assessment of +75M versus +50M in the review | Evidence |
|---|---|---|
| Grammar and fluency | Often locally readable; no consistent step change | Cleaner short zoo/apple episodes coexist with “The read the book,” agreement errors and doubled phrases |
| Logical consistency | Persistent major weakness | Toy-to-cake substitutions, refrigerator self-containment, contradictory intentions and unexplained transitions |
| Character consistency | Some seed-specific repair, clear regressions elsewhere | Fox remains fox in one probe; Eliza→Karen/Zizzy/Eliel and Rose→Emma recur |
| Prompt retention | Mixed; vocabulary retention does not ensure fact retention | +75M crayons return, but toy location remains wrong and music/helping premises dissolve |
| Narrative progression | Some better-contained episodes; no broad problem–attempt–resolution gain established | Zoo visit is improved; nested boxes, search loops and generic activity sequences persist |
| Ending quality | More EOS than +50M, not consistently better resolution | Short happy endings often do not resolve the actual setup; some earlier EOS pairs now cap |
| Repetition and semantic loops | Extreme +50M literal loop shrinks, local loops remain or worsen | Book/1337 worsens; book/2027 remains poor; zero-repeat outputs can still contradict prompts |

## 5. Concrete three-way story evidence

Excerpts are verbatim substrings from the saved continuations, separately quoted without silently stitching omitted material. Each case uses exactly the same prompt ID and seed at all three milestones. Token counts, stop reasons and repetition values describe the complete saved continuation, not just the excerpt.

### 1. A real local improvement: the zoo trip — `dev_03445/3407`

Prompt: Once upon a time, there was a little girl named Lily. She loved animals and wanted to visit the zoo. Her mommy promised to bring her there on

**+25M** (126 tokens; eos; repeated-4-grams 0.0000):

> The storm replied, "Yes, we can go now."

**+50M** (118 tokens; eos; repeated-4-grams 0.0000):

> But suddenly, Lily realized that the truck was very cheap.

> She cried and dropped the truck.

**+75M** (95 tokens; eos; repeated-4-grams 0.0000):

> After the trip, Lily and her mommy went home. Lily was very tired but happy to have a new friend.

The +25M storm answers for a person; +50M abandons the zoo for a truck that Lily tries to carry. +75M stays with the zoo, meets a lion, plays, and returns home. This is a meaningful local improvement in topic continuity and closure. A talking lion is compatible with a children's fantasy; it is not itself an error. The final 'sharing my zoo' wording is still awkward.

Source rows: [+25M line 84](25m/stories.jsonl#L84); [+50M line 84](50m/stories.jsonl#L84); [+75M line 84](75m/stories.jsonl#L84).

### 2. A smaller repair: Sarah remains Sarah — `dev_07496/2027`

Prompt: Once there was a girl named Sarah. She was three years old and loved to dance. On a sunny day, Sarah found an object that she really wanted

**+25M** (141 tokens; eos; repeated-4-grams 0.0072):

> "Look, Mom! I'm going to go and dance!" Lily said.

**+50M** (148 tokens; eos; repeated-4-grams 0.0414):

> Her mom smiled and said, "That's great, Sarah. You can wear it when you put it in your pocket."

**+75M** (75 tokens; eos; repeated-4-grams 0.0000):

> Sarah kept dancing until she felt very tired. She ran home and hugged it tight.

At +25M Sarah is briefly called Lily. At +50M an object turns into a bird that the mother says can be worn and put in a pocket. +75M identifies an apple and keeps the Sarah/dancing thread without a new protagonist name. This is cleaner local composition, although hugging the apple and the sudden fear are weakly motivated. It is not evidence of deeper problem solving.

Source rows: [+25M line 128](25m/stories.jsonl#L128); [+50M line 128](50m/stories.jsonl#L128); [+75M line 128](75m/stories.jsonl#L128).

### 3. Fox identity improves in one seed, not uniformly — `manual_1/3407`

Prompt: Once upon a time, a little fox found a blue box.

**+25M** (148 tokens; eos; repeated-4-grams 0.0759):

> The fox and the fox played together and had lots of fun.

**+50M** (230 tokens; eos; repeated-4-grams 0.0352):

> The little rabbit thought for a moment and said, "I will open it and see what is inside."

**+75M** (142 tokens; eos; repeated-4-grams 0.0072):

> The fox went home and told his friends about his adventure.

> They all said, "I found a treasure for you! I made it for me!"

The +25M fox talks to a duplicated fox. +50M loses the fox to rabbit/bear roles and contradictory box states. +75M retains the fox through a recognisable treasure/home sequence. Speaker perspective and gift ownership still fail in the quoted dialogue. The other two seeds do not show a clean solution: +75M/1337 repeats nested boxes; +75M/2027 introduces a dog friend. This is a seed-specific improvement, not mastery of the probe.

Source rows: [+25M line 153](25m/stories.jsonl#L153); [+50M line 153](50m/stories.jsonl#L153); [+75M line 153](75m/stories.jsonl#L153).

### 4. Largely unchanged: a simple party — `dev_10607/2027`

Prompt: Once upon a time, there were two friends. They liked to do things together. One day, they had a new idea. They decided to join the

**+25M** (43 tokens; eos; repeated-4-grams 0.0000):

> They would run and jump and play games. They laughed and had so much fun!

**+50M** (43 tokens; eos; repeated-4-grams 0.0000):

> They played and laughed and sang songs. They had so much fun. They were very happy.

**+75M** (83 tokens; eos; repeated-4-grams 0.0125):

> They would run and jump and play games. They laughed and had so much fun!

All three deliver a readable, generic friends-at-a-party fragment. +75M adds sunset, hugs and bedtime, but little additional causal development. It is longer here (83 versus 43 tokens at each earlier gate), despite the overall length reduction. This is approximately unchanged qualitative capability, not literally identical output.

Source rows: [+25M line 14](25m/stories.jsonl#L14); [+50M line 14](50m/stories.jsonl#L14); [+75M line 14](75m/stories.jsonl#L14).

### 5. Persistent prompt-fact failure despite clean termination — `manual_3/1337`

Prompt: Tom put his toy under the bed. Later, he looked for it and

**+25M** (137 tokens; eos; repeated-4-grams 0.0149):

> found it under a bush.

**+50M** (69 tokens; eos; repeated-4-grams 0.0000):

> found it under a bush.

**+75M** (64 tokens; eos; repeated-4-grams 0.0000):

> found it under a bush.

The exact prompt says Tom put his toy UNDER THE BED; there is no intervening move. All three instead retrieve it under a bush. The +75M story has 64 continuation tokens, EOS termination and zero repeated token 4-grams. Those favourable diagnostics do not repair the explicit location contradiction. Across all three seeds at every gate, this probe lacks a clear, causally consistent under-bed retrieval; the other seeds drift into rocks, boxes, ownership and unrelated play.

Source rows: [+25M line 157](25m/stories.jsonl#L157); [+50M line 157](50m/stories.jsonl#L157); [+75M line 157](75m/stories.jsonl#L157).

### 6. Persistent unearned resolution: helping a friend — `manual_2/3407`

Prompt: Lily wanted to help her friend, but she did not know how.

**+25M** (132 tokens; eos; repeated-4-grams 0.0000):

> She helped her find the tree to climb up the tree.

**+50M** (87 tokens; eos; repeated-4-grams 0.0595):

> "Sure, we can help you make the laundry together."

**+75M** (97 tokens; eos; repeated-4-grams 0.0106):

> Lily felt a little better and went home with her new toy. She was happy that she could help her friend.

The prompt asks how Lily can help her friend. The earlier outputs drift into climbing or laundry; +75M solves Lily's own desire for a toy and then asserts that she helped her friend. The necessary friend's problem–action–resolution chain is absent. The other two +75M seeds similarly redirect help toward Lily or a bike rather than establishing the friend's need.

Source rows: [+25M line 156](25m/stories.jsonl#L156); [+50M line 156](50m/stories.jsonl#L156); [+75M line 156](75m/stories.jsonl#L156).

### 7. A clear regression from the cleaner +50M garden story — `dev_09383/3407`

Prompt: Once upon a time there was a little girl named Rose. She loved to explore the world around her. One day, she decided to explore a new garden

**+25M** (155 tokens; eos; repeated-4-grams 0.0000):

> She quickly ran towards the dog and asked her to help him.

**+50M** (101 tokens; eos; repeated-4-grams 0.0102):

> Rose was so happy, she jumped up and down. She had so much fun exploring the garden.

**+75M** (145 tokens; eos; repeated-4-grams 0.0141):

> Emma smiled and felt proud of herself for being so kind and generous.

+25M has confused dog/pronoun interactions. +50M maintains Rose, garden, flowers and a butterfly in a relatively contained episode. +75M asks Rose to grab an implausibly heavy tree, introduces a crying mother, then renames Rose to Emma and awards generosity with no demonstrated generous act. This is a direct counterexample to monotonic qualitative improvement.

Source rows: [+25M line 36](25m/stories.jsonl#L36); [+50M line 36](50m/stories.jsonl#L36); [+75M line 36](75m/stories.jsonl#L36).

### 8. Name stability regresses again at +75M — `dev_01251/1337`

Prompt: Once upon a time there was a little girl named Eliza. She loved to play outside and she was very graceful. She had a red ball that she

**+25M** (153 tokens; eos; repeated-4-grams 0.0400):

> One day, Judy was playing in the park when she saw a little girl with a blue ball.

**+50M** (99 tokens; eos; repeated-4-grams 0.0104):

> One day, she wanted to play catch with her ball.

**+75M** (161 tokens; eos; repeated-4-grams 0.0253):

> One day, Karen was playing in the garden when she saw a little girl with a blue ball.

> Zizzy got lost and sad.

> The next day, Eliel went back to playing and the little girl was very sad.

The prompt names Eliza. +25M drifts through Judy, Buzz, Angie and Lizzy. +50M avoids that particular series of protagonist renamings, though its ball/bird actions remain confused. +75M returns to Karen/Zizzy/Eliel without introducing them as separate people. Lower development CE has not stabilised this character representation.

Source rows: [+25M line 106](25m/stories.jsonl#L106); [+50M line 106](50m/stories.jsonl#L106); [+75M line 106](75m/stories.jsonl#L106).

### 9. A +75M reading loop worsens while the story gets shorter — `dev_01952/1337`

Prompt: Once upon a time, there was a mom. She had a special book. The book was a novel. The novel was very tough. The mom wanted

**+25M** (72 tokens; eos; repeated-4-grams 0.0145):

> The novel was so happy again. The end.

**+50M** (94 tokens; eos; repeated-4-grams 0.0330):

> The book had pictures of animals, animals, and animals.

**+75M** (82 tokens; eos; repeated-4-grams 0.1266):

> They read the book. The read the book. It was very interesting.

> They read the book again.

All versions are weak: +25M gives the novel unexplained agency; +50M repeats animals and hugs the book. +75M intensifies the reading loop and adds an overt grammatical error. Its repeated-4-gram fraction is 0.1266 versus 0.0330 at +50M, despite falling from 94 to 82 tokens. EOS therefore conceals a worse local loop, not a well-earned ending.

Source rows: [+25M line 1](25m/stories.jsonl#L1); [+50M line 1](50m/stories.jsonl#L1); [+75M line 1](75m/stories.jsonl#L1).

### 10. The worst +50M loop is reduced, but not solved — `dev_01952/2027`

Prompt: Once upon a time, there was a mom. She had a special book. The book was a novel. The novel was very tough. The mom wanted

**+25M** (107 tokens; eos; repeated-4-grams 0.0962):

> The novel was so happy. It learned how to read it.

**+50M** (214 tokens; eos; repeated-4-grams 0.3318):

> The frog was not a frog. It was a frog. It was a frog. It was a frog.

**+75M** (108 tokens; eos; repeated-4-grams 0.1524):

> She read the novel. It was very yummy.

> The story teaches us a valuable lesson: it is important to be careful when you get lost.

This is the one additional diagnostic-selected triplet, beyond the systematic review set: it had +50M's highest repetition. The severe read/frog loop shrinks at +75M (214 to 108 tokens; repetition 0.3318 to 0.1524), but +75M still exceeds +25M's 0.0962 repetition, shifts toward an unexplained little girl/bookcase, treats the novel as yummy, and closes with an unrelated lost-person moral. Tail repair is real, overall story coherence is still poor.

Source rows: [+25M line 2](25m/stories.jsonl#L2); [+50M line 2](50m/stories.jsonl#L2); [+75M line 2](75m/stories.jsonl#L2).

### 11. No ending repair: hide-and-seek still runs into the cap — `dev_06045/2027`

Prompt: Tom and Lily were playing hide and seek in the big house. Tom was the seeker and Lily was the hider. Lily ran to the living room and saw

**+25M** (300 tokens; max_tokens; repeated-4-grams 0.0471):

> Tom hiding behind a bush.

**+50M** (300 tokens; max_tokens; repeated-4-grams 0.0471):

> Tom hiding behind a bush.

**+75M** (300 tokens; max_tokens; repeated-4-grams 0.0471):

> Tom hiding behind a tree.

> But the kitchen was not happy.

The prompt explicitly makes Tom the seeker and Lily the hider. All three reverse that role immediately. Every version of this seed reaches 300 tokens without EOS; +75M later changes the acting entity to the kitchen and wanders into cake dialogue, ending mid-sentence. All three +75M seeds for this prompt cap at 300. Fewer capped stories in the aggregate has not solved this hard prompt.

Source rows: [+25M line 146](25m/stories.jsonl#L146); [+50M line 146](50m/stories.jsonl#L146); [+75M line 146](75m/stories.jsonl#L146).

### 12. Prompt vocabulary improves while identity and ending regress — `dev_10452/3407`

Prompt: Tom and Mia like to play with crayons. They have many colors of crayons. They like to mark on paper and make pictures. Sometimes they mark on the

**+25M** (300 tokens; max_tokens; repeated-4-grams 0.1010):

> He wants to write a story about a lion.

**+50M** (240 tokens; eos; repeated-4-grams 0.0928):

> "Look, Mia, I found a magic ring. It is a magic ring. You can draw on it and make a wish."

**+75M** (300 tokens; max_tokens; repeated-4-grams 0.0976):

> "Sure, Sara. I will help you make your own pictures. I am happy for you."

> It is not very pretty. It is rare. And your paper is beautiful.

The crayon/drawing prompt becomes a lion-writing story at +25M and a magic-ring story at +50M. +75M uses crayons and paper more directly, but renames Mia to Sara, contradicts itself about the picture, and ends at the 300-token cap in an open quotation. +50M ended with EOS at 240 tokens. The case is mixed on topic retention and regresses on completion; it cannot be scored as a simple improvement.

Source rows: [+25M line 105](25m/stories.jsonl#L105); [+50M line 105](50m/stories.jsonl#L105); [+75M line 105](75m/stories.jsonl#L105).

### 13. Low-level object semantics remain broken — `dev_10185/2027`

Prompt: Once there was a beautiful refrigerator. It was so shiny and white that everybody wanted it. 
One day, a family wanted to buy it. So

**+25M** (93 tokens; eos; repeated-4-grams 0.0111):

> they went to the store and bought some yummy cake.

**+50M** (112 tokens; eos; repeated-4-grams 0.0183):

> the family put the refrigerator in the refrigerator.

**+75M** (106 tokens; eos; repeated-4-grams 0.1068):

> they bought it and put it in the fridge.

The prompt concerns buying a refrigerator. +25M drops it for cake; +50M retains the noun but puts the refrigerator inside itself. +75M still places the just-bought object in a fridge, with no other purchase introduced, and repeats generic happiness. Its repetition rises from 0.0183 to 0.1068. Topic words alone are not object-state consistency.

Source rows: [+25M line 26](25m/stories.jsonl#L26); [+50M line 26](50m/stories.jsonl#L26); [+75M line 26](75m/stories.jsonl#L26).

### 14. A modest cooking repair with residual schema errors — `dev_09121/1337`

Prompt: Once upon a time, there was a boy named Timmy. Timmy loved to help his mom in the kitchen. One day, his mom asked him to stir

**+25M** (133 tokens; eos; repeated-4-grams 0.0462):

> Later that day, Timmy's mom poured some soup into a bowl of soup.

**+50M** (137 tokens; eos; repeated-4-grams 0.0448):

> he always wore a spoon when he cooked.

**+75M** (139 tokens; eos; repeated-4-grams 0.0441):

> He knew he was good at stirring the batter and making a cake.

+75M keeps a batter–stirring–cake thread more consistently than +25M's soup/cake switches or +50M's bread/spoon misuse. However, cooking sugar for dinner and repeating the instruction to stir after the batter is ready remain weak. This is a local gain in semantic organisation, not a mature narrative arc.

Source rows: [+25M line 28](25m/stories.jsonl#L28); [+50M line 28](50m/stories.jsonl#L28); [+75M line 28](75m/stories.jsonl#L28).


## 6. Weaknesses that survive lower development loss

### Stable characters and world state

Characters are not reliably tracked across a paragraph: Rose/Emma, Eliza/Karen/Zizzy/Eliel, Glad/Gadlor, Boo/Bree and Freddy/Benny appear in the reviewed +75M outputs. Objects also lose type, location or ownership. These errors remain possible even when most individual sentences are grammatical and next-token choices are more probable on average.

### Causal structure and earned endings

“Happy,” “proud,” “brave,” hugs and friendship often replace a demonstrated solution. In the helping probe, the friend is not given a specific need that Lily then meets. In box/tree/key stories, an obstacle is introduced and then an opening, surprise or new friend appears without the action needed to resolve it.

A sentence such as “She was happy that she could help her friend” is not evidence that the preceding narrative showed help. Likewise, EOS after a generic moral is not an earned ending.

### Prompt facts beyond shared vocabulary

The toy-location probe is a compact failure, so its error cannot be attributed solely to long output. Other +75M stories preserve words such as crayons, refrigerator or spoon while misusing their properties, referents or actions. Repeating a prompt noun is a weaker achievement than preserving the proposition involving that noun.

### Semantic loops beyond literal 4-grams

The generator can repeatedly search, open boxes, climb or congratulate characters while changing enough tokens to avoid exact 4-gram duplication. Conversely, ordinary repeated phrases are not automatically bad stories. The quantitative metric therefore complements close reading but cannot label semantic coherence.

### Limited long-range planning

At 300 generated tokens, some continuations exceed the 256-token context and the current generation implementation retains only the latest window. This is a plausible constraint on long-horizon retention, not a diagnosis proved by these artifacts. It does not explain why the under-bed fact is contradicted in the first generated words or why character names drift in shorter stories.

The data do not identify a single cause. Capacity, context, training objective, tokenization, decoding and data distributions are hypotheses requiring controlled experiments; no architecture/LR/source change is justified by this comparison alone.

## 7. Overall assessment of +75M

### Is additional pretraining improving generation?

It is clearly improving the **recorded predictive scores**. It also yields some recognisable local narrative improvements: the zoo episode is more coherent, the Sarah fragment simpler and one fox story more identity-stable. These gains are real examples from matched outputs.

However, improved likelihood is not the same measurement as story coherence. The review finds no consistent repair of character/state tracking, prompt-fact retention or causal endings, and supplies direct +75M regressions. The shorter outputs and repaired worst-case +50M loop should not be interpreted as a general increase in storytelling ability.

### Is +75M meaningfully better than +50M?

**Not established as a broad qualitative claim.** A fair description is: **better language modelling, somewhat more concise generations, mixed storytelling**. Some individual examples meaningfully improve; several regress and the simple manual tests remain unreliable. The evidence is insufficient to call +75M a decisive storytelling upgrade or to quantify a human preference advantage.

This is not a finding that +75M is globally worse either. The review is unblinded and covers 60/159 triplets; the remaining seed variants can change the balance. Blank ratings provide no independent human validation. A blinded, matched, multi-rater review would be the appropriate measurement for a stronger aggregate quality claim, not an automatic extra training run.

### Diminishing returns or overfitting?

- **Diminishing marginal predictive gains:** a smaller dev-CE reduction over the second equal 25M-position interval supports this limited endpoint observation. It does not establish a plateau, optimal stopping point or expected benefit from another interval.
- **Qualitative diminishing returns:** the repeated failure modes despite 50M additional positions beyond +25M suggest that additional exposure alone is not reliably fixing these specific generation capabilities. This remains a qualitative observation, not a calibrated learning-curve estimate.
- **Overfitting:** no obvious aggregate signal in these saved windows: both train and dev CE fall, and their gap is nearly stable. Individual story regressions are not evidence of overfitting by themselves.
- **Unresolved:** these artifacts contain no independent training replicas, external held-out test evaluation, paired per-window losses, memorization/nearest-neighbour analysis or filled human ratings. Repeated use of the same development set also limits how strongly it can support final generalization claims.

**Final assessment:** retain +75M as a successfully measured milestone with improved recorded CE/perplexity and mixed generation results. Its strongest story examples are worth preserving, but +75M has not demonstrated reliable prompt/state reasoning or a meaningful overall qualitative advantage over +50M. This report performs no further experiment.

## Appendix A. Input hashes and preservation

These nine files were read and hashed at audit start. All existing source/report/evaluation files included in the preservation baseline are rechecked after report creation. No evaluation output, rating cell or previous report is edited.

| Milestone | Input file, repository-relative | SHA256 |
|---|---|---|
| +25M | `stage_6_tinystories/evaluations/25m/evaluation.json` | `01636d61d9c3513646875c3c9b807167a8826680b6e8c696b83fba80f4e2e328` |
| +25M | `stage_6_tinystories/evaluations/25m/ratings.csv` | `e1dff27e0befa056c730c5cddf25195bc72a7c03d9e34107ab36642e5952b311` |
| +25M | `stage_6_tinystories/evaluations/25m/stories.jsonl` | `afc00350a948345115e6d07bacc4f399d3bdb60017a6eebabc5070b73c4e8426` |
| +50M | `stage_6_tinystories/evaluations/50m/evaluation.json` | `701abe2886ad8186087a22f1f0865e6586789af1ac96e16f24443586b09dc343` |
| +50M | `stage_6_tinystories/evaluations/50m/ratings.csv` | `e740a0d9110bb5006fb9f66e1a98d314596f70486da397a0e7d2ec2c41c6b94f` |
| +50M | `stage_6_tinystories/evaluations/50m/stories.jsonl` | `1b23f68f7c739e4324fb5ce6bd1f51f49fe5429ef1c1ba2a7238b28aff85737c` |
| +75M | `stage_6_tinystories/evaluations/75m/evaluation.json` | `c2bb682df25cbc1865980ea0a64441cebd350873f909f194d26102a4718f074b` |
| +75M | `stage_6_tinystories/evaluations/75m/ratings.csv` | `99db1d432217d86a77f659191c3ca2a0aa7142742cbc82dcbf76cfb7f3191c59` |
| +75M | `stage_6_tinystories/evaluations/75m/stories.jsonl` | `6c9bb467a757bb2701fe1087241ae880e167bfdc22a313c10bd7235f7bf4f156` |

Canonical report: `stage_6_tinystories/evaluations/COMPARISON_THROUGH_75M.md`.
The provenance reports remain historical context; their earlier access limitations
are not silently rewritten here.

## Appendix B. Close-reading coverage ledger

These are assistant observations, not human scores. Each row refers to a matched three-milestone triplet. The first 59 rows follow the deterministic coverage procedure; the final book/2027 row is the explicitly selected repetition-tail diagnostic.

| Prompt ID | Seed | Brief observation on +75M and its earlier counterparts |
|---|---:|---|
| `dev_00041` | 1337 | Still duplicates the old man and confuses speakers; new fish does not create coherent causality. |
| `dev_00188` | 2027 | Shorter jog story removes the threatening-man detour, but actions and ending remain illogical. |
| `dev_00405` | 3407 | Introduces a second Lucy, then an unexplained toy-to-ball change. |
| `dev_01251` | 1337 | Karen/Zizzy/Eliel drift returns; no sustained identity repair. |
| `dev_01323` | 2027 | Dirty-train problem changes to expense/repair, then the train flies away. |
| `dev_01470` | 3407 | Balloon episode is simpler than +50M, but beach setting is lost. |
| `dev_01607` | 1337 | Much shorter than +50M; paper-to-picnic transition remains unexplained. |
| `dev_01682` | 2027 | Pine tree remains inedible/unstable; chase, gift and nest do not connect. |
| `dev_01815` | 3407 | More stable toy episode and ending; problem/mother dialogue is weak. |
| `dev_01952` | 1337 | Reading loop intensifies and includes 'The read the book.' |
| `dev_02309` | 2027 | Stone storyline replaces other incoherence; pronouns and resolution remain weak. |
| `dev_03445` | 3407 | Clearer zoo–lion–home arc than both earlier versions. |
| `dev_03659` | 1337 | Garage mystery becomes nested-box loop; improves neither prompt resolution nor causality. |
| `dev_03932` | 2027 | Frog/wolf/rabbit identities and threat/help roles remain unstable. |
| `dev_04121` | 3407 | Girl calls herself a dog; missing-mother dialogue occurs while mother is present. |
| `dev_04708` | 1337 | Cloud topic returns but Lily gains paws and cloud actions stay incoherent. |
| `dev_04714` | 2027 | Girl/pony/pet/tail confusion persists; names drift. |
| `dev_04788` | 3407 | Sword setup gives way to a bear hugging a bear without resolving the prince's task. |
| `dev_05064` | 1337 | Key/lock/box states contradict; happy ending does not resolve the obstacle. |
| `dev_05150` | 2027 | Rose-sharing play is cleaner than +50M's flying episode, with residual self-address error. |
| `dev_05236` | 3407 | Shorter note episode still has nested-container nonsense; music premise is lost. |
| `dev_05579` | 1337 | Some dancing returns, but a second Lily and unexplained dough derail the episode. |
| `dev_05698` | 2027 | Toy/game picnic remains generic; the wise man's helping role is not developed. |
| `dev_05845` | 3407 | Cake becomes ice cream; man/girl roles remain confused. |
| `dev_05958` | 1337 | Gold transforms into wand/flower; no meaningful treasure progression. |
| `dev_06045` | 2027 | All three cap at 300; hide/seek roles and entities remain unstable. |
| `dev_06168` | 3407 | Freddy becomes Benny; running and rescue logic remain unresolved. |
| `dev_06510` | 1337 | Names are more stable than +50M, but neighbour conflict is displaced by a tree. |
| `dev_06893` | 2027 | Boat episode stays near beach topic; 'wore her boat' and spatial loops remain. |
| `dev_07228` | 3407 | Snowman topic returns, but cake/hot/cold phrase breaks semantic continuity. |
| `dev_07369` | 1337 | Ball is briefly retained, then bird dialogue and Lily flying displace it. |
| `dev_07496` | 2027 | Sarah stays named and the apple/dance ending is simpler; plot remains thin. |
| `dev_08034` | 3407 | Long-name corruption reduced to Gen, but toy-to-cake substitution persists. |
| `dev_08044` | 1337 | Boo/Bree drift and speaker confusion persist despite a clearer butterfly scene. |
| `dev_08235` | 2027 | Climbing loop replaced by incoherent rock scene and Glad-to-Gadlor drift. |
| `dev_08387` | 3407 | Vase/candy episode has contradictory motives and unearned lesson; not a clear gain. |
| `dev_08416` | 1337 | Voice/farmer threat ends with gratitude and Fluffy without a causal bridge. |
| `dev_08818` | 2027 | Shorter fish episode removes +50M's hospital ending, but care premise is lost. |
| `dev_08978` | 3407 | Disaster resolves through an unexplained village journey; pronouns and agency unclear. |
| `dev_09121` | 1337 | Batter/cake focus improves locally; sugar-for-dinner and repeated stirring remain odd. |
| `dev_09136` | 2027 | Spelling is finally mentioned, then dissolves into movement/friendship filler. |
| `dev_09383` | 3407 | Regresses from +50M's garden focus to Rose/Emma and an unearned moral. |
| `dev_09463` | 1337 | Packing/trip changes to hospital; circular packing and arrival remain. |
| `dev_09612` | 2027 | Buying a bear ends in eating it; object category error persists. |
| `dev_09850` | 3407 | Tree/hole/toy actions do not connect; generic caution moral. |
| `dev_10066` | 1337 | Car scene becomes work/race then sudden dog threat; no stable resolution. |
| `dev_10185` | 2027 | Refrigerator self-containment remains; repeated-4-gram tail worsens. |
| `dev_10452` | 3407 | More crayon vocabulary, but Mia/Sara drift, contradictions and capped ending regress versus +50M. |
| `dev_10568` | 1337 | Car/driver distinction worsens; abstract zip ending remains illogical. |
| `dev_10607` | 2027 | Same simple party/friendship formula; longer filler does not add a richer arc. |
| `manual_1` | 3407 | Fox retained more consistently than +50M rabbit/bear drift; treasure exchange still has speaker problems. |
| `manual_2` | 1337 | Helping friend becomes helping Lily ride a colour-changing bike. |
| `manual_3` | 2027 | Toy search becomes loan/closet scene; original under-bed fact not resolved. |
| `manual_1` | 1337 | Fox is more stable than +50M, but nested-box play repeats without payoff. |
| `manual_1` | 2027 | Fox eventually gains a dog friend without setup; box/toy roles still loose. |
| `manual_2` | 2027 | Max helps Lily rather than Lily solving a friend's stated problem. |
| `manual_2` | 3407 | Lily gets her own toy; claimed help for friend is unsupported. |
| `manual_3` | 1337 | Same under-bush contradiction survives; short EOS/zero repetition is not prompt retention. |
| `manual_3` | 3407 | Search loses toy ownership; mother's sharing ending is not a retrieval solution. |
| `dev_01952` | 2027 | Worst +50M frog loop is reduced, but +75M still loops and ends with an unrelated lost-person moral. |
