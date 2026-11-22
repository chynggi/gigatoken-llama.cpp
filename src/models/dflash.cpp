#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act == "swigluoai") {
            hparams.llm_ffn_op = LLM_FFN_SWIGLU_OAI_MOE;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;
    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    if (has_markov) {
        variant = VARIANT_DSPARK;
    } else if (has_selector) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    // The drafter reuses a decoder block from the target family. `decoder_arch` names it;
    // when absent the block is the Qwen3-style default (separate Q/K/V, no output gate).
    // Poolside's DFlashLaguna* drafters set it to "laguna": fused QKV, a softplus output
    // gate, and QK-norm -- see conversion notes in this file's load_arch_tensors.
    {
        std::string decoder_arch;
        if (ml.get_key(LLM_KV_DFLASH_DECODER_ARCH, decoder_arch, false)) {
            if (decoder_arch == "laguna") {
                hparams.dflash_decoder_laguna = true;
            } else if (decoder_arch != "qwen3") {
                throw std::runtime_error("unsupported DFlash decoder_arch: '" + decoder_arch + "'");
            }
        }
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    // per-tap width of the target features, as for EAGLE3: n_embd by default, n_streams*n_embd for a hyper-connection target
    uint32_t n_embd_tgt = hparams.n_embd;
    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt, false);
    if (n_embd_tgt == 0) {
        throw std::runtime_error("DFlash 'target_hidden_size' must be positive");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s], n_embd_tgt = %u\n", __func__, layers.c_str(), n_embd_tgt);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern, hparams.n_layer(), false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:  load_tensors_dflash2();             break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DFLASH:                                       break;
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: one RMSNorm per extracted target layer, applied to each aux hidden state
    // *before* they are concatenated and fused by fc (HF: model.aux_hidden_norms.{i}).
    // Exports stack them into a single [n_embd, n_target_layers] tensor.
    output_norm_enc_aux = create_tensor(tn(LLM_TENSOR_ENC_AUX_NORM, "weight"),
            { n_embd, (int64_t) target_layer_ids.size() }, TENSOR_NOT_REQUIRED);

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }
}

uint32_t llama_model_dflash::graph_nodes_per_tensor() const {
    if (hparams.dsv4_hc_mult > 0) {
        return 32;
    }
    if (hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft.
        // The selector lattice scales with the draft block size; 12u was calibrated
        // on block-8 drafters, so scale the budget for larger blocks (e.g. Ripple's 16).
        return 12u*std::max<uint32_t>(1u, hparams.dflash_block_size/8u);
    }
    return llama_model_base::graph_nodes_per_tensor();
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    // Laguna-family backbone (poolside DFlashLaguna* drafters): fused QKV, a softplus
    // attention output gate, QK-norm, dense SwiGLU MLP. Mirrors llama_model_laguna's
    // dense layers; the drafter has no MoE (num_experts = 0).
    if (hparams.dflash_decoder_laguna) {
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            const int64_t n_head_il    = hparams.n_head(i);
            const int64_t n_head_kv_il = hparams.n_head_kv(i);
            const int64_t n_embd_q_il  = n_embd_head_k * n_head_il;
            const int64_t n_embd_k_il  = n_embd_head_k * n_head_kv_il;
            const int64_t n_embd_v_il  = n_embd_head_v * n_head_kv_il;

            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

            create_tensor_qkv(layer, i, n_embd, n_embd_q_il, n_embd_k_il, n_embd_v_il, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_q_il, n_embd }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

            // Gate width selects per-head vs per-element, exactly as in llama_model_laguna.
            // The XS.2 drafters are per-head ("gating": "per-head" -> g_proj -> n_head).
            const int64_t n_gate_per_head = n_head_il;
            const int64_t n_gate_per_elem = n_embd_head_k * n_head_il;
            const ggml_tensor * gate_meta = ml->get_tensor_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str().c_str());
            int64_t n_gate_out = n_gate_per_head;
            if (gate_meta != nullptr) {
                n_gate_out = gate_meta->ne[1];
                if (n_gate_out != n_gate_per_head && n_gate_out != n_gate_per_elem) {
                    GGML_ABORT("DFlash/laguna: unexpected attention gate width %lld at layer %d "
                               "(expected %lld per-head or %lld per-element)",
                               (long long) n_gate_out, i, (long long) n_gate_per_head, (long long) n_gate_per_elem);
                }
            }
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, n_gate_out }, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        }
        return;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

// target features wider than taps x draft n_embd (the raw residual streams of a hyper-connection target) are not normalized, so fc(x) can exceed the F16 range
// build_lora_mm is mul_mat(fc, x), optionally followed by mul(., fc_s) and add(., lora): request F32 on every matmul of that chain, since each reads the raw features
static void dflash_fc_prec_f32(ggml_tensor * t, int & n_set) {
    if (t == nullptr) {
        return;
    }
    switch (t->op) {
        case GGML_OP_MUL_MAT:
            n_set += ggml_prec_set_acc(t, GGML_PREC_F32);
            dflash_fc_prec_f32(t->src[1], n_set); // lora: b * (a * x)
            break;
        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SCALE:
            dflash_fc_prec_f32(t->src[0], n_set);
            dflash_fc_prec_f32(t->src[1], n_set);
            break;
        default:
            break;
    }
}

static ggml_tensor * build_dflash_fc(const llm_graph_context & g, const llama_model & model, ggml_tensor * inp) {
    ggml_tensor * cur = g.build_lora_mm(model.fc, inp, model.fc_s);
    // only when the fused target features are wider than taps x draft n_embd; n_embd-wide taps keep the default precision
    if (g.hparams.n_embd_inp_enc() > model.target_layer_ids.size() * g.hparams.n_embd) {
        int n_set = 0;
        dflash_fc_prec_f32(cur, n_set);
        GGML_ASSERT(n_set > 0 && "DFlash fc: no matmul found to request F32 accumulation");
    }
    return cur;
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    // Normalize each extracted target-layer feature with its own RMSNorm before fusing.
    // `cur` is the concatenation [layer_0 | layer_1 | ... ] along the embedding axis, so
    // reshaping to [n_embd, n_target_layers, n_tokens] lines the slices up with the
    // stacked [n_embd, n_target_layers] weight, which broadcasts over the token axis.
    if (model.output_norm_enc_aux) {
        const int64_t n_embd_aux = model.output_norm_enc_aux->ne[0];
        const int64_t n_aux      = model.output_norm_enc_aux->ne[1];

        GGML_ASSERT(cur->ne[0] == n_embd_aux * n_aux);

        cur = ggml_reshape_3d(ctx0, cur, n_embd_aux, n_aux, n_tokens);
        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.output_norm_enc_aux);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_aux * n_aux, n_tokens);
        cb(cur, "enc_aux_norm_out", -1);
    }

    cur = build_dflash_fc(*this, model, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    const int64_t n_embd_inp = g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = build_dflash_fc(g, model, inp_target);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) {
    if (!hparams.has_rope(il)) {
        return cur;
    }

    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const int64_t n_embd_kv_il = n_embd_head * n_head_kv_il;
        const bool shared_kv = !layer.wqkv && layer.wv == nullptr;

        ggml_tensor * Kcur;
        ggml_tensor * Vcur;

        if (layer.wqkv) {
            // Fused QKV (laguna backbone). Injection runs over every accepted token, so
            // slice the K/V row block out of the weight instead of projecting Q and
            // throwing it away. Rows are contiguous, so the view stays a valid mul_mat
            // operand for quantized weights too.
            const int64_t n_embd_q_il = layer.wqkv->ne[1] - 2*n_embd_kv_il;

            ggml_tensor * wkv = ggml_view_2d(ctx0, layer.wqkv,
                    layer.wqkv->ne[0], 2*n_embd_kv_il,
                    layer.wqkv->nb[1], n_embd_q_il*layer.wqkv->nb[1]);

            ggml_tensor * kv = ggml_mul_mat(ctx0, wkv, inp_g);

            Kcur = ggml_view_3d(ctx0, kv, n_embd_head, n_head_kv_il, n_tokens,
                    ggml_row_size(kv->type, n_embd_head), kv->nb[1], 0);
            Vcur = ggml_view_3d(ctx0, kv, n_embd_head, n_head_kv_il, n_tokens,
                    ggml_row_size(kv->type, n_embd_head), kv->nb[1],
                    ggml_row_size(kv->type, n_embd_kv_il));
        } else {
            Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, inp_g, layer.wk_s), n_embd_head, n_head_kv_il, n_tokens);
            Vcur = shared_kv ? Kcur : ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, inp_g, layer.wv_s), n_embd_head, n_head_kv_il, n_tokens);
        }

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        if (use_iswa) {
            // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            // rotate K/V into the cache's rotated space
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
            if (k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        } else {
            // rotate K/V into the cache's rotated space
            if (inp_attn->self_k_rot[il]) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
            }
            if (inp_attn->self_v_rot[il]) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inpL = ggml_get_rows(ctx0, get_tok_embd(*this, model), inp_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        const int64_t n_head_il    = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const bool shared_kv = !layer.wqkv && layer.wv == nullptr;

        ggml_tensor * Qcur;
        ggml_tensor * Kcur;
        ggml_tensor * Vcur;

        if (shared_kv) {
            // Gemma4-style backbone without wv: V shares the K projection
            Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, noise_norm, layer.wq_s), n_embd_head, n_head_il,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, noise_norm, layer.wk_s), n_embd_head, n_head_kv_il, n_tokens);
            Vcur = Kcur;
        } else {
            // handles both the Qwen3-style separate Q/K/V and the laguna-style fused QKV
            auto qkv = build_qkv(layer, noise_norm, n_embd_head, n_head_il, n_head_kv_il, il);
            Qcur = qkv.q;
            Kcur = qkv.k;
            Vcur = qkv.v;
        }

        // laguna backbone: softplus attention output gate, projected from the same
        // pre-attention hidden state that feeds q/k/v (matches the HF reference)
        ggml_tensor * gate = nullptr;
        if (layer.wqkv_gate) {
            gate = build_lora_mm(layer.wqkv_gate, noise_norm);
            cb(gate, "attn_gate_proj", il);
        }

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention. When the layer is gated, o_proj is deferred
        // until after the gate is applied.
        ggml_tensor * const wo   = gate ? nullptr : layer.wo;
        ggml_tensor * const wo_s = gate ? nullptr : layer.wo_s;

        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, wo, NULL, wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      wo, NULL, wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (gate) {
            // per-head gate (g_proj -> n_head) broadcasts over head_dim; per-element
            // (g_proj -> n_head*head_dim) multiplies the attention output directly
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            if (layer.wqkv_gate->ne[1] == n_head_il) {
                cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head, n_head_il, n_tokens);
                gate = ggml_reshape_3d(ctx0, gate, 1,           n_head_il, n_tokens);
                cur  = ggml_mul(ctx0, cur, gate);
                cur  = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head_il, n_tokens);
            } else {
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
            cb(cur, "attn_o_proj", il);
        }

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    return cur;
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:  return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
