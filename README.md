<h2 align="center">
  Official Code for the paper: <b>M</b>ultilingual <b>E</b>valuation of <b>G</b>enerative <b>A</b>I (MEGA), a framework to evaluate Large Language Models (LLMs) on various multilingual benchmarks
</h2>

<p align="center">
  <a href="https://2023.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2023-blue"></a>
  <a href="https://github.com/microsoft/MMLMCalibration/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>


Generative AI models have shown impressive performance on many Natural Language Processing tasks such as language understanding, reasoning, and language generation. An important question being asked by the AI community today is about the capabilities and limits of these models, and it is clear that evaluating generative AI is very challenging. Most studies on generative LLMs have been restricted to English and it is unclear how capable these models are at understanding and generating text in other languages. We present the first comprehensive benchmarking of generative LLMs - MEGA, which evaluates models on standard NLP benchmarks, covering 16 NLP datasets across 70 typologically diverse languages. We compare the performance of generative LLMs including Chat-GPT and GPT-4 to State of the Art (SOTA) non-autoregressive models on these tasks to determine how well generative models perform compared to the previous generation of LLMs. We present a thorough analysis of the performance of models across languages and tasks and discuss challenges in improving the performance of generative LLMs on low-resource languages. We create a framework for evaluating generative LLMs in the multilingual setting and provide directions for future progress in the field.


# Getting Started

### Dependencies
- Compatible with Python3.7
- The necessary packages can be install through requirements.txt.

### Setup

We recommend creating a virtual environment for the framework (optional):

```shell
$ [sudo] pip install virtualenv
$ virtualenv -p python3 megaenv
$ source megaenv/bin/activate
```

Install the required packages by running:


```shell
pip install -r requirements.txt
```

The framework also requires setting up the [promptsource](https://github.com/bigscience-workshop/promptsource/tree/eval-hackathon). To set up follow the steps below:

```shell
cd promptsource
pip install -e .
```

Finally, the framework requires keys for [OpenAI API](https://platform.openai.com) and [Azure Translation API](https://www.microsoft.com/en-us/translator/business/translator-api/). Place the keys in the [`keys/`](keys/) directory with the filenames `openai_key.txt` and `bing_translate_key.txt`.

# Overview of the Framework
We use prompt-based approach to evaluate LLMs on multilingual benchmark. For each task we divide our prompt into two portions: i) Few-shot Labelled examples and ii) Test Example to evaluate. The few-shot examples and the test example can be in different languages and we denote the language used to represent few-shot examples as *Pivot Language* and *Target Language* to represent the test example. Taking inspiration from [Shi et al. 2022](https://arxiv.org/abs/2210.03057), we consider three setups for evaluation:
1. Zero-Shot Cross Lingual Prompting: Where the pivot and target languages are different
2. Monolingual Prompting: Pivot and target languages are same
3. Translated Test: Test example is translated to pivot language before evaluating

The figures below provide examples of the three setups:

![Zero-Shot Prompting](images/zero_shot_prompting.jpg)
![Monolingual Prompting](images/monolingual_prompting.jpg)
![Translate Test](images/translate_test.jpg)


# Running Evaluation

## XNLI

The following language prompts are currently available in the framework

Language | Available Prompts
-------- | -----------------
en (English) | GPT-3 Style, MNLI crowdsource, always/sometimes/never, based on the previous passage,...
hi (Hindi) | English GPT-3 Style, Handcrafted GPT-3 Style, English based on the previous passage, Handcrafted based on the previous passage, ...

The full list can be found in `promptsource`. Check [`promptsource/README.md`](promptsource/README.md) for details.

To run the evaluation on XNLI, execute the following command
```shell
$ python -m mega.eval_xnli \
    -t {Target Language} \
    -p {Pivot Language} \
    -k {Number of Few-shot Examples} \
    --tgt_prompt_name {Prompt Name For Target} \ 
    --pivot_prompt_name {Prompt Name For Pivot} \
    --model {GPT Model to Evaluate} \
    {--translate-test}
```

An example command would look like:

```shell
python -m mega.eval_xnli \
    -p hi \
    -t hi \
    -k 4 \
    --pivot_prompt_name "Handcrafted based on the previous passage" \
    --tgt_prompt_name "Handcrafted based on the previous passage" \
    --model DaVinci003
```

**Other tasks to be added soon!**

## Extending the framework to workf for other Classification/QA tasks:

Extending to other classification and QA tasks is simple. First create the prompts for different languages for a selected task using promptsource (next section). Then we only need to load the dataset for the task and the prompt templates to perform evaluation. Check sample code below for PAWS-X:
```python
# Import the necessary modules to run evaluation
from mega.eval.eval_cls import evaluate_model
from mega.data.data_utils import choose_few_shot_examples

# Import datasets and promptsource libraries
from datasets import load_dataset
from promptsource.templates import DatasetTemplates


# Load dataset of your choice
dataset = "paws-x"
src_lang = "en" #Can change the language from en to the language of your choice 
tgt_lang = "en" #Similarly language here can be changed, if it is same as src_lang then monolingual, else zero-shot
train_dataset = load_dataset(dataset, src_lang)["train"] 
test_dataset = load_dataset(dataset, tgt_lang)["test"]

# Load prompt templates for the dataset
prompt_name = "Meaning" # Name of the prompt created by you on promptsource
train_prompt = DatasetTemplates(f"{dataset}/{src_lang}")[prompt_name]
test_prompt = DatasetTemplates(f"{dataset}/{tgt_lang}")[prompt_name]

# Run evaluation
accuracy = evaluate_model(
        train_dataset,
        test_dataset,
        train_prompt,
        test_prompt,
        model="DaVinci003", #Can change this to BLOOM also
        few_shot_size=4, #Number of few-shot examples
        save_preds_path="results/preds.csv",#Any path where you would like to store predictions,
        temperature=0.1, # Temperature parameter for GPT-3x generations
    )
print(accuracy)
```


# Creating New Prompts
Adapted from [`promptsource/README.md`](promptsource/README.md)

PromptSource provides a Web-based GUI that enables developers to write prompts in a templating language and immediately view their outputs on different examples.

There are 3 modes in the app:
- **Sourcing**: create and write new prompts
- **Prompted dataset viewer**: check the prompts you wrote (or the existing ones) on the entire dataset
- **Helicopter view**: aggregate high-level metrics on the current state of P3

![ALT](promptsource/assets/promptsource_app.png)

To launch the app locally, please first make sure you have followed the steps in [Setup](#setup), and from the root directory of the repo, run:
```bash
cd promptsource
streamlit run promptsource/app.py
```

You can also browse through existing prompts on the [hosted version of PromptSource](https://bigscience.huggingface.co/promptsource). Note the hosted version disables the Sourcing mode (`streamlit run promptsource/app.py -- --read-only`).

### Writing prompts
Before creating new prompts, you should read the [contribution guidelines](CONTRIBUTING.md) which give an step-by-step description of how to contribute to the collection of prompts.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
