trl vllm-serve --model Qwen/Qwen2.5-3B --tensor_parallel_size 1 --enable_prefix_caching True --gpu_memory_utilization 0.95 --host 0.0.0.0 --port 8000
