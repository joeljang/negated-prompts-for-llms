python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation" --model opt-125m
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation" --model opt-350m
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation" --model opt-1.3b
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation" --model opt-2.7b
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation" --batch 1 --model opt-6.7b

python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation negation2" --model opt-125m
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation negation2" --model opt-350m
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation negation2" --model opt-1.3b
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation negation2" --model opt-2.7b
python score.py --dataset imdb --promptsource --sample 200 --prompt_name "review generation negation2" --batch 1 --model opt-6.7b