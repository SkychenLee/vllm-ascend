export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7


export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1/0 是否打屏
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_HOST_LOG_FILE_NUM=1000

# 使用 FIA - Triton
export VLLM_ASCEND_USE_PAGED_ATTENTION=1
# 使用 MXFP4 量化
export VLLM_ASCEND_PAGED_ATTN_USE_MXFP4_P=1
# 使用 三阶 HIF4
export VLLM_ASCEND_PAGED_ATTN_USE_HIF4_P=0
# 使用 一阶 HIF4
export VLLM_ASCEND_PAGED_ATTN_USE_HIF4_ONCE_P=0

# 对 KV 进行 一次性窗口 伪量化
export VLLM_ASCEND_ENABLE_KV_WINDOW_QUANT=1
# 开启 ATTENTION SINK
export VLLM_ASCEND_ENABLE_ATTENTION_SINK=1
# Tail 高精窗口大小
export VLLM_ASCEND_HIGH_PRECISION_WINDOW_SIZE=128
# Sink 窗口大小
export VLLM_ASCEND_ATTENTION_SINK_SIZE=128

# 开启 QK 旋转
export VLLM_ASCEND_ENABLE_QK_ROTATION=1
export VLLM_ASCEND_ROT_H_PATH="./block_rht_matrix.pt"

# 开启 HIF4 量化
export VLLM_ASCEND_ENABLE_HIF4=0


vllm serve /PATH \
       --host 0.0.0.0 \
       --served-model-name Qwen3-div \
       --trust-remote-code \
       --port 10125 \
       --gpu-memory-utilization 0.9 \
       --block-size 128 \
       --distributed-executor-backend mp \
       --no-enable-prefix-caching \
       --async-scheduling \
       --max-model-len 40960 \
       --max-num-batched-tokens 40960 \
       --max-num-seqs 400 \
       --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false}}' \
       --quantization ascend \
       --tensor-parallel-size 4 \
       --enable-expert-parallel \
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
