CUDA_VISIBLE_DEVICES=0 python src/get_completions_vllm.py \
    --model_path /scr/kanishkg/morals/results/Qwen3-1.7B-Base-v2/best-ckpt \
    --save_dir /scr/kanishkg/morals/results/Qwen3-1.7B-Base-v2/ \
    --max_tokens 1024 \
    --temperature 0.0 \
    --top_p 0.95
