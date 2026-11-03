# Generation compatibility source

`generation_compat_model.py` is the only retained compatibility variant. It has
SHA256 `b058a3621e93103100e017742bc15a08cc9a586f750b59df5b93fdc1a42417d3`.

The active `../model.py` is the checkpoint-compatible training source and must
remain unchanged for resume identity checks. The compatibility variant changes
the PAD/BOS logit assignments used during generation on the legacy CUDA runtime;
evaluation wrappers used it as a temporary source copy. It is not a newer training
model and must not replace the active source.
