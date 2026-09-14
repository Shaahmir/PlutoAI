from config import CONFIG

def parameter_count():

    embedding_parameters = CONFIG.VOCAB_SIZE * CONFIG.D_MODEL

    q_proj = CONFIG.D_MODEL * (CONFIG.N_HEADS * CONFIG.HEAD_DIM)
    kv_proj = 2 * CONFIG.D_MODEL * (CONFIG.N_KV_HEADS * CONFIG.HEAD_DIM)
    o_proj = CONFIG.D_MODEL * CONFIG.D_MODEL
    attention_parameters = q_proj + kv_proj + o_proj

    swiglu_parameters = 3 * CONFIG.D_MODEL * CONFIG.D_FF # Gate + up + down
    rmsnorm_parameters = 2 * CONFIG.D_MODEL # 2 RMSNorms per transformer block

    block_parameters = attention_parameters + swiglu_parameters + rmsnorm_parameters
    transformer_parameters = CONFIG.N_LAYERS * block_parameters
    final_layer_norm = CONFIG.D_MODEL

    lm_head_parameters = 0 if CONFIG.USE_WEIGHT_TYING else CONFIG.VOCAB_SIZE * CONFIG.D_MODEL
    
    return embedding_parameters + transformer_parameters + final_layer_norm + lm_head_parameters

def main():

    print("="*50)
    print("Model Confgiguration: ")
    print()

    print(f"Vocabulary Size:  {CONFIG.VOCAB_SIZE:,}")
    print(f"CONTEXT LENGTH :  {CONFIG.CONTEXT_LENGTH:,}")
    print(f"Layers:  {CONFIG.N_LAYERS:,}")
    print(f"Hidden Size:  {CONFIG.D_MODEL:,}")
    print(f"Attention Heads:  {CONFIG.N_HEADS:,}")
    print(f"Head Dimension:  {CONFIG.HEAD_DIM:,}")
    print(f"FFN Size:  {CONFIG.D_FF:,}")
    print(f"Estimated Parameters: {parameter_count() / 1e6:.2f}M")

if __name__ == "__main__":
    main()
