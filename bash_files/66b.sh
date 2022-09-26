CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset ai2_arc --dataset_config ARC-Easy --promptsource --sample 200 --batch 8 --prompt_name "q&a" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset ai2_arc --dataset_config ARC-Easy --promptsource --sample 200 --batch 8 --prompt_name "q&a negation" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset super_glue --dataset_config copa --promptsource  --batch 8 --prompt_name "generate" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset super_glue --dataset_config copa --promptsource  --batch 8 --prompt_name "generate negation" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset hellaswag --promptsource --sample 200 --batch 8 --prompt_name "Open-ended completion" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset hellaswag --promptsource --sample 200 --batch 8 --prompt_name "Open-ended completion negation" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset story_cloze --dataset_config 2016 --promptsource --sample 200 --batch 8 --prompt_name "Generate Ending" --model opt-66b
CUDA_VISIBLE_DEVICES=8,9,10,11 python score.py --dataset story_cloze --dataset_config 2016 --promptsource --sample 200 --batch 8 --prompt_name "Generate Ending Negation" --model opt-66b