# Done

-evaluation of {GPT, T0} with {promptsource dataset, original dataset}

    How to use promptsource dataset?:
    original:
    python score.py --dataset rte --model T0_3B
    using promptsource:
    python score.py --dataset glue --dataset_config qqp --promptsource --prompt_name "quora" --model T0_3B

-multibatch available for T0, but not for GPT.

-evaluation length = min(500, length of the dataset)
exceptionally for glue/qqp: 200
(see data_loaders.py l 769~774)

-You can add other T0 models in score.py l 63
(Note: You should have "T0" in your model, and only bigscience model is available, such as bigscience/T0-> name as T0, bigscience/T0pp-> name as T0pp.
models like google/t5-v1_1-xl are not available)

-parallelize not implemented yet

## Dependencies

You can use `pip install -r requirements.txt` to install the required libraries.

## OpenAI Beta
To use GPT-3 you must use OpenAI Beta, which is limited access. You can apply for access [here](https://beta.openai.com/). Once you have access you will need to point the `score.py` to your API key with the `--key` argument or put your key in `api.key` which is the default path. 

## Downloading Datasets

`DATA_README.md` has thorough instructions for downloading and processing datasets. We provide automatic downloaders and processers for datasets where possible in `data_downloaders/` but see `DATA_README` for full instructions.

## Running Scorers
Once you have a dataset downloaded, running all the zero-shot scoring strategies at once is as simple as:

```
python score.py <dataset abbrevation> --model <model>
```

where `<dataset-abbreviation>` is the abbreviation for a given dataset used for table rows in the paper. If there is any confusion, simply look in `score.py` to see how dataset selection works. `<model>` is the name of either a GPT-2 or GPT-3 model e.g. `xl`, `davinci`, etc. To speed things up you can use a larger `--batch` if you have enough GPU memory.
