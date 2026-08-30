#include "models.h"

void llama_model_k2_horizon::load_arch_hparams(llama_model_loader & ml) {
    // generic
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups, false);

    hparams.f_norm_group_eps = hparams.f_norm_rms_eps;
    if (hparams.n_norm_groups == 0) hparams.n_norm_groups = 1;
    
    // moe
    if (hparams.n_expert > 0) {
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
        ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);
        ml.get_key(LLM_KV_MOE_EVERY_N_LAYERS, hparams.moe_every_n_layers, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
        if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        }
    }

    // mova
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT, hparams.n_value_expert, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, hparams.n_value_expert_used, false);
    if (hparams.n_value_expert > 0) {
        GGML_ASSERT(hparams.n_value_expert <= LLAMA_MAX_EXPERTS);
        GGML_ASSERT(hparams.n_value_expert_used > 0);
        GGML_ASSERT(hparams.n_value_expert_used <= hparams.n_value_expert);
    }
    else {
        GGML_ASSERT(hparams.n_value_expert_used == 0);
    }

    // model size info
    if (hparams.n_layer() == 28 && hparams.n_embd == 1536) {
        type = LLM_TYPE_1B;
    }
    else if (hparams.n_layer() == 48 && hparams.n_embd == 2560) {
        type = LLM_TYPE_36B;
    }
    else {
        type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_k2_horizon::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS; // initializing variables basically

    // embeddings
    tok_embd = create_tensor(
        tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
        {n_embd, n_vocab},
        0
    );

    // final norm and output projection
    output_norm = create_tensor(
        tn(LLM_TENSOR_OUTPUT_NORM, "weight"),
        {n_embd},
        0
    );

    // output
    output = create_tensor(
        tn(LLM_TENSOR_OUTPUT, "weight"),
        {n_embd, n_vocab},
        TENSOR_NOT_REQUIRED // can be tied with embedding (indicated by tensor not found in .gguf). see next conditional
    );
    if (output == nullptr) {
        output = create_tensor(
            tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
            {n_embd, n_vocab},
            TENSOR_DUPLICATED
        );
    }

    for (int i = 0; i < n_layer; i++){
        auto & layer = layers[i];
        const bool is_moe_layer = n_expert > 0 && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0; // in the architecture, if mova is moe as well
        
        // attn normalization
        layer.attn_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_NORM, "weight", i),
            {n_embd},
            0
        );
        
        // query and key tensors, always dense. and their optional normalization
        // query
        layer.wq = create_tensor(
            tn(LLM_TENSOR_ATTN_Q, "weight", i),
            {n_embd, n_embd_head_k * n_head},
            0
        );
        layer.attn_q_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i),
            {n_embd_head_k * n_head},
            TENSOR_NOT_REQUIRED
        );

        // key
        layer.wk = create_tensor(
            tn(LLM_TENSOR_ATTN_K, "weight", i),
            {n_embd, n_embd_k_gqa},
            0
        );
        layer.attn_k_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i),
            {n_embd_k_gqa},
            TENSOR_NOT_REQUIRED
        );

        // value tensors, possible MoVA
        if (is_mova_layer) {
            layer.attn_v_gate = create_tensor(
                tn(LLM_TENSOR_ATTN_V_GATE, "weight", i),
                {n_embd, hparams.n_value_expert},
                0
            );
            layer.attn_v_gate_b = create_tensor(
                tn(LLM_TENSOR_ATTN_V_GATE, "bias", i),
                {hparams.n_value_expert},
                TENSOR_NOT_REQUIRED
            );
            layer.attn_v_exps = create_tensor(
                tn(LLM_TENSOR_ATTN_V_EXPS, "weight", i),
                {n_embd, n_embd_v_gqa, hparams.n_value_expert},
                0
            );
        }
        else {
            layer.wv = create_tensor(
                tn(LLM_TENSOR_ATTN_V, "weight", i),
                {n_embd, n_embd_v_gqa},
                0
            );
        }

        // attn output projection
        layer.wo = create_tensor(
            tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_embd_head_v * n_head, n_embd},
            0
        );

        // optional softplus gate
        layer.wqkv_gate = create_tensor(
            tn(LLM_TENSOR_ATTN_GATE, "weight", i),
            {n_embd, n_embd_head_v * n_head},
            TENSOR_NOT_REQUIRED
        );

        // FFN normalization
        layer.ffn_norm = create_tensor(
            tn(LLM_TENSOR_FFN_NORM, "weight", i),
            {n_embd},
            0
        );

        // MoE stuff
        if (is_moe_layer) {
            if (hparams.n_ff_exp == 0){
                throw std::runtime_error("K2 MoE layer requires expert_feed_forward_length");
            }
            
            // moe router and it's optional bias
            layer.ffn_gate_inp = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_expert},
                0
            );
            layer.ffn_exp_probs_b = create_tensor(
                tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i),
                {n_expert},
                TENSOR_NOT_REQUIRED
            );

            // routed experts (up, gate, and down)
            layer.ffn_up_exps = create_tensor(
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),
                {n_embd, hparams.n_ff_exp, n_expert},
                0
            );
            layer.ffn_gate_exps = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                {n_embd, hparams.n_ff_exp, n_expert},
                0
            );
            layer.ffn_down_exps = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {hparams.n_ff_exp, n_embd, n_expert},
                0
            );

            // shared experts (always evaluated)
            if (hparams.n_expert_shared > 0) {
                int64_t n_ff_shexp;
                if (hparams.n_ff_shexp > 0) {
                    n_ff_shexp = hparams.n_ff_shexp;
                } else {
                    n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
                }

                // up gate down
                layer.ffn_up_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i),
                    {n_embd, n_ff_shexp},
                    0
                );
                layer.ffn_gate_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i),
                    {n_embd, n_ff_shexp},
                    0
                );
                layer.ffn_down_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i),
                    {n_ff_shexp, n_embd},
                    0
                );
            }
        }
        else {
            // ordinary up gate down
            layer.ffn_up = create_tensor(
                tn(LLM_TENSOR_FFN_UP, "weight", i),
                {n_embd, n_ff},
                0
            );
            layer.ffn_gate = create_tensor(
                tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_embd, n_ff},
                0
            );
            layer.ffn_down = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN, "weight", i),
                {n_ff, n_embd},
                0
            );
        }
        
    }

}