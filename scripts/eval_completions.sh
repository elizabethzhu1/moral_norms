echo "Evaluating Qwen3-1.7B-Base-v3"
CUDA_VISIBLE_DEVICES=0 python src/get_completions_vllm.py \
    --model_path  Qwen/Qwen3-1.7B-Base \
    --save_dir /scr/kanishkg/morals/results/Qwen3-1.7B-Base-v3/checkpoint-0 \
    --max_tokens 1024 \
    --temperature 0.2 \
    --top_p 0.95

for n in $(seq 10 10 600); do
    echo "Evaluating Qwen3-1.7B-Base-v3 checkpoint-${n}"
    CUDA_VISIBLE_DEVICES=0 python src/get_completions_vllm.py \
        --model_path /scr/kanishkg/morals/results/Qwen3-1.7B-Base-v3/checkpoint-${n} \
        --save_dir /scr/kanishkg/morals/results/Qwen3-1.7B-Base-v3/checkpoint-${n} \
        --max_tokens 1024 \
        --temperature 0.2 \
        --top_p 0.95
done