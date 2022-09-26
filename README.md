# Can Large Language Models Truly Understand Prompts? A Case Study with Negated Prompts

![alt text](figure1.png "Example of Negated Prompt")

We aim to answer four main questions in this work. (1) How does scaling the size of LMs affect their abilities to understand the concept of negation? (2) Are LMs explicitly trained to follow instructions (InstructGPT) better at understanding negated instructions? (3) Can In-Context Learning or Fine-tuning help mitigate this problem? (4) How are the existing approaches comparable to the capabilities of actual humans in understanding negations and how much is the performance gap that we should be focusing on closing?

The answers can be found in our draft! Come check it out! :) 

## Replication 

To replicate the results of our paper, use the following commands

## Dependencies

You can use `pip install -r requirements.txt` to install the required libraries.

## OpenAI Beta
To use GPT-3 you must use OpenAI Beta, which is limited access. You can apply for access [here](https://beta.openai.com/). Once you have access you will need to point the `score.py` to your API key with the `--key` argument or put your key in `api.key` which is the default path. 

## Running Scorers
Once you have a dataset downloaded, running all the zero-shot scoring strategies at once is as simple as:

```
python score.py <dataset abbrevation> --model <model>
```

where `<dataset-abbreviation>` is the abbreviation for a given dataset used for table rows in the paper. If there is any confusion, simply look in `score.py` to see how dataset selection works. `<model>` is the name of either a GPT-2 or GPT-3 model e.g. `xl`, `davinci`, etc. To speed things up you can use a larger `--batch` if you have enough GPU memory.
