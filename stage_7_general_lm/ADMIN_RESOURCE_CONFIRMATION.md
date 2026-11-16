# Stage 7/8 — resource questions for the HPC administrator

Prepared 4 November 2026. This is a checklist for the user; no message has been sent or HPC query executed.

Known Stage 6 evidence: PBS requested 4 GB host RAM; Tesla V100 GPUs had 32 GB VRAM; home-directory corpus/checkpoints persisted between jobs. The separate “32 GB per job” figure is unverified. Please provide values, units, applicable queues/paths and policy sources.

| Area | Exact details to confirm |
|---|---|
| Account/queue | Which project/account and GPU queue may this user use? Is the old Stage 6 project/queue still valid? What exact PBS resource-selector syntax is required? |
| Host RAM/CPUs | Default and maximum RAM per job/node; CPU allowance; whether PBS `mem` is per node or aggregate; units and enforcement. May one GPU request 20 GB host RAM, or is 4 GB a ceiling? Which job-accounting fields report actual peak RAM? |
| GPU allocation | Current V100 32GB availability for this account; maximum GPUs per job/user and concurrent jobs; exclusive versus shared assignment; any partitioning; exact selector guaranteeing GPU model and accessible VRAM. |
| Walltime/interruption | Hard walltime for the selected queue/project; whether 64 hours is allowed; extension policy; preemption/requeue behavior; termination warnings and grace interval; permitted dependent continuation jobs. |
| Aggregate compute | Remaining GPU-hour/service-unit allocation and expiry; charging formula; daily/weekly limits; concurrency and fair-share restrictions. |
| Home storage | Filesystem/path; current used bytes; soft/hard byte quota; inode usage/limits; grace periods; shared-project accounting; whether home is approved/recommended for training I/O. |
| Persistent project storage | Available mount path, current use and allocation, inode limits, permissions, retention/expiry, backup/snapshot/restore policy, and availability across compute nodes/jobs. |
| Scratch/temporary storage | Exact node-local and shared paths; per-job/user byte and inode limits; whether capacity is reserved; purge triggers/deadlines; survival after job/node termination; guaranteed copy-out period, if any. |
| Filesystem behavior/I/O | On the proposed checkpoint path, support/limitations for hard links, atomic replacement and file/directory fsync; durability guidance; I/O throttles and maximum file sizes. Administrator guidance still needs later target-filesystem validation. |
| Runtime/software | Supported compute-node Python/PyTorch/CUDA and driver versions; permission to create venv/conda environments or containers; approved installation/cache paths and container runtime; recommended compatible Transformers/PEFT stack. |
| Downloads/staging | Permitted staging node/path; outbound download policy, proxy or approved mirrors; size, licensing and security-review restrictions for datasets/model weights. |

Record each answer as **confirmed**, **unknown**, or **not permitted**, with date and scope. Recheck free capacity before admission because shared usage changes. Filesystem-wide free space does not establish user quota headroom. Persistence does not establish backup coverage.

Minimum answers before a benchmark launch: valid queue/account/resource syntax; permitted CPU/RAM and target GPU; short-job walltime/charge allowance; sufficient confirmed probe/report capacity on an approved filesystem; compatible permitted runtime. Corpus packing, pretrained downloads and main training require their additional quota, retention and aggregate-allocation answers first.

No password or other secret is needed. This checklist grants no permission to submit, transfer, install, or modify existing artifacts.
