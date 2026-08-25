#!/usr/bin/env python3
"""Rewrite a third-party DFlash draft GGUF into llama.cpp's native `dflash` layout.

Some DFlash drafters are published as GGUFs produced by other forks, which name the
architecture after the target family (e.g. `qwen35-dflash-draft`) and keep the HF
tensor names under a `dflash.` prefix. llama.cpp only loads the `dflash` architecture
with the tensor names emitted by conversion/qwen.py, so this script renames the
metadata and tensors instead of requiring a re-conversion from the HF checkpoint.

Draft GGUFs carry no vocabulary of their own -- the drafter shares the target's
tokenizer and embedding matrix -- so the tokenizer metadata is copied from the target
GGUF that the draft will be paired with.

Usage:
    python convert_dflash_draft_gguf.py DRAFT.gguf TARGET.gguf OUT.gguf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in __import__("os").environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))

import gguf  # noqa: E402

logger = logging.getLogger("convert-dflash-draft-gguf")

# hparams that carry over verbatim once the arch prefix is rewritten
CARRIED_SUFFIXES = (
    "context_length",
    "embedding_length",
    "block_count",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "vocab_size",
    "rope.freq_base",
    "rope.dimension_count",
    "logit_scale",
    "final_logit_softcapping",
    "embedding_scale",
)

# tensors whose HF-ish names differ from what src/models/dflash.cpp looks up
TENSOR_RENAMES = {
    "dflash.fc.weight": "fc.weight",
    "dflash.hidden_norm.weight": "enc.output_norm.weight",
}

# tokenizer metadata is not part of the draft; take it from the target
TOKENIZER_PREFIXES = ("tokenizer.",)


def field_value(field: gguf.ReaderField):
    val = field.contents()
    if isinstance(val, list) and len(val) == 1:
        return val[0]
    return val


def add_kv(writer: gguf.GGUFWriter, key: str, field: gguf.ReaderField) -> None:
    """Re-emit a reader field under `key`, preserving its GGUF value type."""
    val = field_value(field)
    ftype = field.types[0]

    if ftype == gguf.GGUFValueType.ARRAY:
        sub = field.types[1]
        if sub == gguf.GGUFValueType.STRING:
            writer.add_array(key, [s.decode("utf-8") if isinstance(s, bytes) else s for s in val])
        else:
            writer.add_array(key, list(val))
        return

    if ftype == gguf.GGUFValueType.STRING:
        writer.add_string(key, val.decode("utf-8") if isinstance(val, bytes) else str(val))
    elif ftype == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(val))
    elif ftype in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
        writer.add_float32(key, float(val))
    else:
        writer.add_uint32(key, int(val))


def convert(draft_path: Path, target_path: Path, out_path: Path) -> None:
    draft = gguf.GGUFReader(draft_path, "r")
    target = gguf.GGUFReader(target_path, "r")

    src_arch = field_value(draft.fields["general.architecture"])
    if isinstance(src_arch, bytes):
        src_arch = src_arch.decode("utf-8")
    logger.info("source architecture: %s", src_arch)

    if src_arch == "dflash":
        logger.warning("source is already the native `dflash` architecture -- nothing to rename")

    writer = gguf.GGUFWriter(out_path, "dflash")

    # --- hparams -----------------------------------------------------------
    for suffix in CARRIED_SUFFIXES:
        field = draft.fields.get(f"{src_arch}.{suffix}")
        if field is not None:
            add_kv(writer, f"dflash.{suffix}", field)

    # head_dim is what the drafter rotates; foreign exports tend to omit it because
    # their loader derives it, while llama.cpp would otherwise fall back to
    # n_embd / n_head and rotate the wrong number of dimensions.
    if f"{src_arch}.rope.dimension_count" not in draft.fields:
        key_len = draft.fields.get(f"{src_arch}.attention.key_length")
        if key_len is None:
            raise ValueError("draft GGUF has neither rope.dimension_count nor attention.key_length")
        n_rot = int(field_value(key_len))
        writer.add_uint32("dflash.rope.dimension_count", n_rot)
        logger.info("derived dflash.rope.dimension_count = %d from attention.key_length", n_rot)

    # --- DFlash-specific keys ---------------------------------------------
    block_size = draft.fields.get(f"{src_arch}.dflash.block_size") or draft.fields.get(f"{src_arch}.block_size")
    if block_size is None:
        raise ValueError("draft GGUF has no dflash block_size")
    writer.add_uint32("dflash.block_size", int(field_value(block_size)))

    layer_ids_field = (
        draft.fields.get(f"{src_arch}.dflash.target_layer_ids")
        or draft.fields.get(f"{src_arch}.target_layers")
    )
    if layer_ids_field is None:
        raise ValueError("draft GGUF has no dflash target layer ids")
    layer_ids = [int(v) for v in field_value(layer_ids_field)]
    # llama.cpp indexes the *input* to a layer, so an id of k in HF terms (the output
    # of layer k) is extracted as layer input k + 1 -- the same +1 that
    # conversion/qwen.py applies.
    extract_ids = [i + 1 for i in layer_ids]
    writer.add_array("dflash.target_layers", extract_ids)
    logger.info("target_layer_ids %s -> dflash.target_layers %s", layer_ids, extract_ids)

    # These drafters reuse the Laguna decoder block (fused QKV + softplus output gate).
    # Detect it from the tensors rather than trusting a name, then tell the loader.
    has_gate = any(t.name.endswith("attn_gate.weight") for t in draft.tensors)
    if has_gate:
        writer.add_string("dflash.decoder_arch", "laguna")
        logger.info("attn_gate tensors present -> dflash.decoder_arch = laguna")

    # --- tokenizer, copied from the target --------------------------------
    n_tok = 0
    for key, field in target.fields.items():
        if key.startswith(TOKENIZER_PREFIXES):
            add_kv(writer, key, field)
            n_tok += 1
    logger.info("copied %d tokenizer fields from %s", n_tok, target_path.name)

    # --- tensors -----------------------------------------------------------
    aux_norms: dict[int, np.ndarray] = {}
    n_out = 0

    for tensor in draft.tensors:
        name = tensor.name

        if name.startswith("dflash.aux_hidden_norm."):
            idx = int(name.split(".")[2])
            aux_norms[idx] = tensor.data
            continue

        writer.add_tensor(TENSOR_RENAMES.get(name, name), tensor.data, raw_dtype=tensor.tensor_type)
        n_out += 1

    if aux_norms:
        if sorted(aux_norms) != list(range(len(aux_norms))):
            raise ValueError(f"aux_hidden_norm indices are not contiguous: {sorted(aux_norms)}")
        if len(aux_norms) != len(extract_ids):
            raise ValueError(
                f"{len(aux_norms)} aux_hidden_norm tensors but {len(extract_ids)} target layers"
            )
        # llama.cpp reads one stacked [n_embd, n_target_layers] tensor and broadcasts it
        # over the token axis, so the per-layer norms are stacked in extraction order.
        stacked = np.stack([aux_norms[i] for i in range(len(aux_norms))], axis=0)
        writer.add_tensor("enc.aux_norm.weight", stacked, raw_dtype=gguf.GGMLQuantizationType.F32)
        n_out += 1
        logger.info("stacked %d aux_hidden_norm tensors into enc.aux_norm.weight %s",
                    len(aux_norms), tuple(reversed(stacked.shape)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    logger.info("wrote %s (%d tensors)", out_path, n_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("draft", type=Path, help="third-party DFlash draft GGUF to rewrite")
    parser.add_argument("target", type=Path, help="target model GGUF to take the tokenizer from")
    parser.add_argument("output", type=Path, help="path of the rewritten GGUF")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    convert(args.draft, args.target, args.output)


if __name__ == "__main__":
    main()
