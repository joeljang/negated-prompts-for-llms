CUDA_VISIBLE_DEVICES=8 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-125m
CUDA_VISIBLE_DEVICES=8 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-350m
CUDA_VISIBLE_DEVICES=8 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-1.3b
CUDA_VISIBLE_DEVICES=9 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-2.7b
CUDA_VISIBLE_DEVICES=9 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --max_output_length 128 --model T0_3B
CUDA_VISIBLE_DEVICES=10 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-6.7b
CUDA_VISIBLE_DEVICES=11,12 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-13b
CUDA_VISIBLE_DEVICES=13,14 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --max_output_length 128 --model T0
CUDA_VISIBLE_DEVICES=8,9,10 python score.py --dataset lambada --promptsource --sample 300 --prompt_name "please next word negation" --model opt-30b