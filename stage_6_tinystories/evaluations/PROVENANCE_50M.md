# Stage 6 +50M HPC provenance audit

Date: 1 November 2026, Asia/Kolkata.

**Conclusion: PARTIAL — HPC authentication blocked remote verification.** The server was reachable on the retry outside the network sandbox, but SSH authentication was denied. No remote command executed. The +50M evaluation record agrees with the requested checkpoint identity and progress; the actual HPC checkpoint bytes and continuation history remain unverified.

This is an access limitation, not evidence that the checkpoint or training run failed. It is not a PASS for +50M provenance.

## Access attempt and scope

No user SSH configuration was found at `/Users/shash/.ssh/config`. The existing repository instructions identify `cs1230342@hpc.iitd.ac.in`; this documented endpoint was used.

Before each command, its read-only purpose was identified. The remote command was limited to listing the project directory. The connection disabled host-key updates, connection multiplexing, forwarding, and local SSH commands, and retained strict host-key checking and noninteractive authentication:

```sh
ssh -T \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=yes \
  -o UpdateHostKeys=no \
  -o ControlMaster=no -o ControlPath=none \
  -o ForwardAgent=no -o ForwardX11=no \
  -o ClearAllForwardings=yes -o PermitLocalCommand=no \
  -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  cs1230342@hpc.iitd.ac.in \
  'ls -ld -- ~/zenithX9271/NanoGPT'
```

Results:

1. Inside the network sandbox, SSH exited 255 with `Could not resolve hostname hpc.iitd.ac.in: nodename nor servname provided, or not known`.
2. The identical read-only probe was retried outside that sandbox through the approval mechanism. SSH exited 255 with:

   ```text
   cs1230342@hpc.iitd.ac.in: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).
   ```

The second response establishes that the remote SSH server was reached, but authentication did not succeed. It does not establish whether a VPN, different configured alias, SSH agent, or another credential arrangement is needed. No credential installation, host-key change, SSH configuration change, environment repair, alternate connection route, or further authentication attempt was performed.

Subsequent checks only read existing local files, calculate hashes, and deserialize the local +25M checkpoint on CPU with `weights_only=True`. No model was instantiated or executed. The report destination was confirmed absent before creation. This new local report is the only intended write; the earlier comparison report was not edited.

## 1. Checkpoint identity verification

### What the available +50M record establishes

The local [50m/evaluation.json](50m/evaluation.json) records:

| Field | Recorded value | Verification status |
|---|---|---|
| Evaluated checkpoint path | `/home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a/latest.pt` | Read from local evaluation record; current remote path/content unverified. |
| Checkpoint SHA256 | `40d145f6739896f923438e54f5a3235a26bfdb0d5cc136bfafb816871658d04e` | Matches the user's expected identity, but remote checkpoint not rehashed. |
| Stage 6 updates | 6104 | Recorded evaluation counter, not independently read from the HPC checkpoint. |
| Additional positions | 50,003,968 | Matches expectation in evaluation record. |
| Total positions | 148,307,968 | Matches expectation in evaluation record. |
| Total completed updates | Not present in evaluation JSON | 18,104 is expected from 12,000 + 6104; actual checkpoint field unverified. |
| Evaluation status | `complete`; 159/159 generations | Recorded evaluation status, not a training/PBS exit status. |

SHA256 of this local evaluation JSON, recalculated during this phase:

`701abe2886ad8186087a22f1f0865e6586789af1ac96e16f24443586b09dc343`.

According to that record, `latest.pt` was evaluated. Whether `best.pt` or the 50M snapshot contains the same model or identical bytes cannot be established without inspecting the HPC files. A recorded path/hash is not a substitute for rehashing its current contents, particularly because `latest.pt` can change after an evaluation.

### Requested remote checkpoint inventory

Only the latest path below is explicitly named by the available evaluation record. The other paths are candidates derived from the same run directory and naming convention; their existence has not been verified.

| Exact remote path to inspect | Existence | Bytes | Modification time | Actual SHA256 / metadata |
|---|---|---|---|---|
| `/home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a/latest.pt` | Unverified remotely | Unverified | Unverified | Expected evaluation identity above; actual bytes unverified. |
| `/home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a/best.pt` | Unverified | Unverified | Unverified | Unverified; cannot assume best is at update 6104. |
| `/home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a/snapshot_additional_50000000.pt` | Unverified | Unverified | Unverified | Unverified. |
| `/home/cse/btech/cs1230342/zenithX9271/NanoGPT/stage_6_tinystories/runs/s6a/snapshot_additional_25000000.pt` | Unverified remotely | Unverified | Unverified | Relevant potential resume anchor; local copy exists separately. |

Other remote checkpoints could not be enumerated. The shared run directory is a recorded reference, not a live filesystem inventory.

### Local checkpoint anchor, verified directly

The Mac's `stage_6_tinystories/runs/s6a/latest.pt` remains **+25M**:

- Size: **95,591,803 bytes**.
- SHA256: `50f94bf80a142e58fdd0ef3daa933fe4c015c2353a7156d8be288f95cf8f50a6`.
- Stage 6 updates: **3052**.
- Additional positions: **25,001,984**.
- Total positions: **123,305,984**.
- Total completed updates: **15,052**.

It does not match the expected +50M SHA256 and must not be used as evidence that the +50M checkpoint has been recovered or verified. This expected local/remote progress difference is not by itself checkpoint corruption.

## 2. Training continuation verification

| Required assertion | Available evidence | Result |
|---|---|---|
| Resume used the correct +25M checkpoint | Verified local +25M anchor above; no +50M invocation/PBS command available. | **UNVERIFIED** for the actual HPC continuation. |
| Starting resumed update was 3052 | Local prior run reached 3052. No continuation `starting_stage6_updates` event or `resumed_updates` startup output inspected. | **UNVERIFIED**. |
| Training reached update 6104 | +50M evaluation record reports 6104. Remote checkpoint/training log inaccessible. | **PARTIAL:** evaluation-record evidence only. |
| Additional positions reached 50,003,968 | Same value in evaluation record. | **PARTIAL:** evaluation-record evidence only. |
| Total positions reached 148,307,968 | Same value in evaluation record. | **PARTIAL:** evaluation-record evidence only. |
| Total completed updates were 18,104 | Arithmetic expectation only; evaluation JSON lacks this field. | **UNVERIFIED** against actual checkpoint. |
| No invalidating source/config change during continuation | Current local source/config match the local +25M checkpoint. No +50M checkpoint or invocation identities inspected. | **UNVERIFIED** for HPC +50M. |
| Review gate completed normally | Local +25M exit is `review_gate`; no +50M summary, metrics exit, or PBS exit inspected. | **UNVERIFIED** for +50M. |

The available local `runs/s6a/metrics.jsonl` contains 92 events: 2 invocations, 16 checkpoint events, 9 evaluations, 63 training events, and 2 exits. Its maximum recorded Stage 6 update is 3052. Its existing `provenance.json` describes initial Stage 5 initialization, not the requested +25M-to-+50M continuation.

Local source inspection shows why the complete metrics history matters: `train.py` writes `provenance.json` only if it does not already exist, while it appends an `invocation` event on each run. Therefore, even a recovered remote provenance file describing the original invocation would not alone prove or disprove the subsequent resume. The appended metrics and PBS startup output are needed.

## 3. Optimizer, RNG, and schedule verification

The verified local `core.py` implements an exact resume path that:

- Checks complete configuration, data identity, and Python source identities against the checkpoint.
- Loads model weights and calls `optimizer.load_state_dict`.
- Restores run/base counters, saved snapshots, origin, and evaluation state.
- Validates moment shapes/finiteness and optimizer step counters against total completed updates; checks betas and weight decay.
- Restores Python, NumPy, Torch CPU, Torch CUDA, and training-sampler RNG states.
- Computes the next learning rate from the restored cumulative Stage 6 update count and the existing configured schedule.

This is **static implementation evidence from the local +25M-compatible source**, not proof that the inaccessible HPC continuation executed it unchanged.

| Item | Evidence needed from +50M run | Current status |
|---|---|---|
| AdamW moments restored, not reset | Resume invocation/startup, matching source identities, checkpoint moments/counters, and continuation records. Final counters alone would not prove the absence of manual intervention. | **UNVERIFIED**. |
| Optimizer reached total update 18,104 | Step counters and finite, correctly shaped moments in actual +50M optimizer state. | **UNVERIFIED**. |
| RNG restored correctly | Resume-path evidence and recorded source, saved RNG fields, compatible runtime, and training-sampler state. Existence of final RNG state alone would not prove historical restoration. | **UNVERIFIED**. |
| Schedule retained original 100M horizon | Actual +50M stored configuration versus original configuration, invocation arguments, and logged LR values. | **UNVERIFIED**. |
| LR continued without warmup reset | Recorded progression after update 3052 and agreement with the original schedule at the correct pre-update counter. | **UNVERIFIED**. |
| No guard bypass | Remote source/config hashes and executed invocation/log evidence. | **UNVERIFIED**. |

No resume routine, RNG restoration routine, training/evaluation entry point, or optimizer step was invoked during this audit.

## 4. Source and data identity comparison

### Training source reference

These **local** hashes were recalculated and all match the local +25M checkpoint's recorded code identities:

| Stage 6 source | Local SHA256 | Against HPC +50M checkpoint/source |
|---|---|---|
| `__init__.py` | `02f86e3862c8672123220007e6831262abf939756c9fd053907981fb7cd96f89` | Unverified |
| `core.py` | `d339cc0a7bee57197754941734266be25cbaabe75cb1a248fdef1b9394eebe3e` | Unverified |
| `data.py` | `05acdb60f7c8d5bdf94e40de02ac76569299a455ff0243762ee41be11ebdbfaf` | Unverified |
| `evaluate.py` | `f6d3e32336aed91a8e247755f1392ad6ba8ff7fff742853ec41d896810b5ddd1` | Unverified |
| `model.py` | `7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5` | Unverified |
| `train.py` | `9ccb6273e90522da66f1e64b30bdf6cb67d25483e1041a3b859343f66bb3577a` | Unverified |

The local training configuration `configs/continue.json` has SHA256:

`0df15f4064a433c9f6db1678eeefdcdb0edf0de0385c408f5dd3cf1e8223f7b7`.

Its parsed content exactly matches the local +25M checkpoint's stored configuration. Comparison with actual HPC +50M configuration is unverified.

### Original training source versus generation patch

Local recovery archives remain distinct:

- `model.py.original`: `7aca47adbae88c80bdc9d0339dc662761d179f51e5e274715dc9ae00e954cbb5`.
- `model.py.generation_patch.py`: `b058a3621e93103100e017742bc15a08cc9a586f750b59df5b93fdc1a42417d3`.

The difference previously inspected is confined to PAD/BOS logit suppression in `generate`: one combined indexing assignment versus two separate assignments. The active local source now matches the original training identity.

The user reports that the separate generation-only patch was used for the +50M evaluation. That is **user-reported execution context**, consistent with the preserved patch, but it has not been verified against HPC temporary source files, evaluation job logs, or execution-time source hashes. The available evaluation JSON does not contain an execution-time code identity. Consequently:

- Actual +50M checkpoint-recorded training identity: **unverified**.
- Actual temporary evaluation `model.py` identity: **unverified**.
- Restoration of original training source on HPC after evaluation: **unverified**.

A temporary evaluation patch and a training source identity must not be conflated. No checkpoint identity or guard was altered to reconcile them.

### Data references

| Asset | Recalculated local SHA256 | Compared with +50M evaluation record | Compared with actual HPC files/checkpoint |
|---|---|---|---|
| `data/shards/v2/manifest.json` | `89c9df839572ef6a1890caf4b1144229090b8ced8b4ea2c66ed5dd090e17c9a7` | Match | Unverified |
| `data/tokenizer/v2/tokenizer.json` | `99055f35c77392e4df4486685b60fee1de1d95e2c373c26f487cf9944746d038` | Match | Unverified |
| `stage_6_tinystories/configs/evaluation_prompts.json` | `f1c83474faa2b56d670055d6da373a1460eccc007db296bcf509f44f5737429c` | Match | Unverified |

### Architecture

The directly inspected local +25M checkpoint specifies **8 heads, 6 layers, width 256, context 256, vocabulary 8192, dropout 0.2, and tied embeddings**. The prior local audit counted 6,897,152 unique learned parameters. These are the verified architecture references. The actual +50M checkpoint architecture could not be checked; matching recorded evaluation context alone does not establish its full architecture.

## 5. PBS job, log, and metric evidence

The associated +50M PBS job ID, node, startup invocation, stdout/stderr paths, termination status, and scheduler exit status could not be identified or inspected. No scheduler command was run and no job was submitted, cancelled, or otherwise changed.

The local `stage_6_tinystories/hpc/` contains `train.pbs` but no copied +50M job log. Local training metrics stop at +25M. Therefore no intermediate +25M-to-+50M measurements can be reported.

| Measurement | Actual available evidence |
|---|---|
| Last local training entry, update 3052 | Dropout-enabled minibatch loss 2.2807242870; LR `9.070540568360237e-05`; preclip gradient norm 1.4089717865; UTC `2026-10-31T10:01:42Z`. This is the +25M boundary, not a +50M continuation measurement. |
| Last local training exit | `reason=review_gate`, update 3052, +25,001,984 positions; UTC `2026-10-31T10:01:48Z`. This does not establish the +50M exit reason. |
| +50M evaluated train CE | 2.0444016978144646, recorded in standalone evaluation JSON. This is an evaluation-mode train-split score, not the final minibatch training loss. |
| +50M development CE | 2.0949381892569363; recorded perplexity 8.124938746190118. |
| Intermediate development CE after +25M | Unavailable locally; unverified remotely. |
| LR progression after update 3052 | Unavailable locally; unverified remotely. No theoretical LR values are presented as measurements. |
| Training-loss progression after update 3052 | Unavailable locally; unverified remotely. |
| Warnings/errors/restarts/interruptions in continuation | Unverified. Lack of accessible logs must not be interpreted as absence of errors. |
| Final +50M exit reason/PBS status | Unverified. Evaluation `status=complete` is a different event. |

## 6. Unverified items and remaining risks

The central missing evidence is authenticated read-only access to the actual HPC run. Without it, the checkpoint inventory, actual SHA256, progress fields, optimizer/RNG state, source/config identities, continuation invocation, intermediate metrics, generation-patch execution context, and final review-gate/PBS outcome cannot be established.

The practical risks are confusing a current `latest.pt` with the historical checkpoint evaluated at +50M; confusing the local +25M copy with the HPC +50M state; treating the temporary generation patch as the training source; or inferring uninterrupted exact continuation from final counters alone. No discrepancy was repaired and no scientific conclusion was upgraded on the basis of inaccessible records.

To continue this audit, the user needs to make an existing working SSH authentication/session available or identify the existing connection method that should be used. This report does not request or require private key material or passwords to be pasted into chat. No large checkpoint transfer is necessary for the planned in-place read-only audit.

Once access is available and the user authorizes the next action, the remaining work is to stat/hash the remote checkpoint candidates; deserialize metadata in place without model execution or bytecode writes; inspect stored optimizer/RNG/configuration; read the full appended metrics and PBS stdout/stderr; and distinguish checkpoint-recorded training source from the evaluation patch. Those actions were not performed in this phase. Any future report must preserve this artifact rather than overwrite it.

## 7. PASS / PARTIAL / FAIL conclusion

| Scope | Result |
|---|---|
| Local +50M evaluation record agrees with expected hash/counters | **PASS**, as a record-content check only. |
| Local +25M source/config identity reference | **PASS**, directly verified against the local checkpoint. |
| Actual HPC +50M checkpoint identity and architecture | **UNVERIFIED**. |
| HPC training continuation, optimizer/RNG/schedule restoration | **UNVERIFIED**. |
| PBS job/log/normal review-gate completion | **UNVERIFIED**. |
| Overall requested HPC provenance verification | **PARTIAL — blocked by SSH authentication**. |

No evidence was obtained that establishes a checkpoint/provenance failure. Equally, the actual +50M run cannot be certified. Return this limitation to the user and wait for direction; do not resume training, alter sources, bypass guards, or attempt an access repair as part of this audit.
