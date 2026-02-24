---
language:
- eng
pretty_name: OpenHermes 2.5
tags:
- synthetic
- GPT-4
- Distillation
- Compilation
---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/S1OhWCy0EWcvFda4w5w_o.png)

# Dataset Card for Dataset Name

This is the dataset that made OpenHermes 2.5 and Nous Hermes 2 series of models.

Support me on GitHub sponsors <3 : https://github.com/sponsors/teknium1

## Dataset Details

### Dataset Description

The Open Hermes 2/2.5 and Nous Hermes 2 models have made significant advancements of SOTA LLM's over recent months, and are underpinned by this exact compilation and curation of many open source datasets and custom created synthetic datasets.

The Open Hermes 2.5 dataset is a continuation of the Open Hermes 1 dataset, at a much larger scale, much more diverse, and much higher quality compilation, reaching 1M, primarily synthetically generated instruction and chat samples.

## Lilac Integration

This dataset has been pushed to Lilac's (a data curation and exploration platform) live HuggingFace spaces, that hosts many popular OS Datasets for exploration and curation, as well as does Text Embedding searches and Clustering of those datasets

Check out that out here: https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5

## Dataset Sources

### Airoboros 2.2
By Jon Durbin: https://huggingface.co/datasets/jondurbin/airoboros-2.2

### CamelAI Domain Expert Datasets (Physics, Math, Chemistry & Biology)
By CamelAI: https://huggingface.co/camel-ai

### ChatBot Arena (GPT-4 Only)
By LMSys: https://huggingface.co/datasets/lmsys/lmsys-chat-1m

### Collective Cognition (09-11-2023)
By Teknium: https://huggingface.co/datasets/CollectiveCognition/chats-data-2023-09-22

### CoT Alpaca GPT4
I have lost the source page for this dataset, sorry

### Evol Instruct 70K && 140K
By WizardLM: 
70K: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k
140k: https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k

### Glaive Code Assistant
By Sahil & Glaive: https://huggingface.co/datasets/glaiveai/glaive-code-assistant

### GPT4-LLM
By Baolin Peng*, Chunyuan Li*, Pengcheng He*, Michel Galley, Jianfeng Gao
https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned

### GPTeacher
By Teknium & Kuruminha: https://github.com/teknium1/GPTeacher

### Medical Tasks 
By CogStack: https://github.com/CogStack/OpenGPT

### MetaMath 40k
By MetaMath: https://huggingface.co/datasets/meta-math/MetaMathQA

### SlimOrca 550K
By Wing Lian and Guan Wang and Bleys Goodson and Eugene Pentland and Austin Cook and Chanvichet Vong and Teknium:
https://huggingface.co/datasets/Open-Orca/SlimOrca

### Platypus
By Garage-bAInd: https://huggingface.co/datasets/garage-bAInd/Open-Platypus

### ShareGPT (GPT4-Only)
By LMSys: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

### Unnatural Instructions GPT4
By Baolin Peng*, Chunyuan Li*, Pengcheng He*, Michel Galley, Jianfeng Gao:
https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM


## Dataset Structure

The dataset follows a sharegpt structure. This means it is a list of dictionaries, with each dictionary containing a new list of dicts called conversations. Each turn in a conversation has two dictionaries, a "from" field, which denotes the role of that turn, and a "value" field which contains the actual text.

Here is an example of an entry:

```
    {
        "conversations": [
            {
                "from": "system",
                "value": ""
            },
            {
                "from": "human",
                "value": "Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year? Include your logic."
            },
            {
                "from": "gpt",
                "value": "Here's the logic behind this:\n\n1. We know that February has 28 days in a non-leap year.\n2. If the tree drops 7 leaves every day, then over the course of February, it would drop:\n   Leaves dropped in February = Leaves per day * Days in February\n   = 7 leaves * 28 days\n   = 196 leaves\n\nSo, the tree would drop 196 leaves in February in a non-leap year."
            }
        ],
        "source": "airoboros2.2",
        "category": "orca"
    },
```

Some examples, like the one above, contain the metadata that came with the dataset, such as "category" being "orca", and many contain the source dataset that it was curated from. 


## Citation

```bibtex
@misc{OpenHermes 2.5,
  title = {OpenHermes 2.5: An Open Dataset of Synthetic Data for Generalist LLM Assistants},
  author = {Teknium},
  year = {2023},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/teknium/OpenHermes-2.5}
}
```