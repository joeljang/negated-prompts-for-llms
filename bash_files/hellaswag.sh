CUDA_VISIBLE_DEVICES=0 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-125m
CUDA_VISIBLE_DEVICES=0 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-350m
CUDA_VISIBLE_DEVICES=0 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-1.3b
CUDA_VISIBLE_DEVICES=0 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-2.7b
CUDA_VISIBLE_DEVICES=15 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --max_output_length 128 --model T0_3B
CUDA_VISIBLE_DEVICES=1 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-6.7b
CUDA_VISIBLE_DEVICES=12,13 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --max_output_length 128 --model T0
CUDA_VISIBLE_DEVICES=4,5 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-13b
CUDA_VISIBLE_DEVICES=6,7,8 python score.py --dataset hellaswag --promptsource --sample 300 --prompt_name "Open-ended completion" --model opt-30b