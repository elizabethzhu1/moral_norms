CUDA_VISIBLE_DEVICES=0 python src/get_completions_vllm.py \
    --model_path /scr/kanishkg/morals/ \
    --output model_responses.csv \
    --batch_size 32 \
    --max_tokens 512 \
    --temperature 0.0 \
    --top_p 0.95
