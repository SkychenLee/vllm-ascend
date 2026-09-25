# DFlash and DSpark input preparation

`_prepare_dflash_inputs_kernel_ascend` prepares context KV slots, draft query
tokens and positions, sampling indices, and graph padding for MRV2 DFlash and
DSpark. The existing worker patch installs it in the upstream DFlash module.

## Parallel work

The upstream launcher selects a `(request, worker)` grid from the largest
scheduled request. Every worker processes a disjoint contiguous context and
query partition. The earlier Ascend implementation returned immediately from
all workers except worker zero and processed the entire context serially.

Block-table reads remain scalar within each partition because supported
Triton Ascend versions cannot lower clamped vector gathers reliably. Graph
padding uses 256-element vector stores, distributed across every active
program independently of the upstream decode tile size. Thus the unused tail
of a large token capacity is no longer cleared with one scalar store per token.
Only worker zero writes each request's scalar metadata and sampling state.

## Contract

- At least one request is required, as enforced by the upstream wrapper.
- Each request has at least one valid context token after rejection.
- During chunked prefill, rejection counts are zero and the anchor comes from
  `next_prefill_tokens`. Otherwise it comes from `last_sampled`.
- Block-table rows follow scheduled request order; `idx_mapping` selects
  persistent sampling state. These two indices must not be interchanged.
- Null block IDs and rejected context slots map to `PAD_SLOT_ID`.
- Query positions and sequence lengths are clipped at the model limit, while
  sampling positions preserve the upstream unclipped convention.
- Padding uses zero sampling indices and positions, minus-one sampling state
  mappings, and `PAD_SLOT_ID` for query KV slots. Unused query token/position
  buffers and unscheduled sampling states are left untouched.
- Optional CP arguments follow the newer upstream ABI; callers without them
  retain CP=1 behavior. Interleave size must divide the physical block size.

No device-to-host reads, allocations, or TP communication are added. The
kernel changes metadata preparation only; model weights and LoRA arithmetic
are unaffected.

## Validation

`test_prepare_dflash_inputs.py` checks exact agreement with an independent CPU
reference for decode, 8K prefill, uneven chunks, rejection, null blocks,
sampling layouts, CP ownership, clipping and padding. It also exercises the
current upstream wrapper and NPU graph replay with changing request mappings
and positions. These are operator tests, not a full serving performance test.
