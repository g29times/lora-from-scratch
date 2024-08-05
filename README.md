# LoRA from Scratch

LoRA, which stands for Low-Rank Adaptation, is a popular technique to finetune LLMs more efficiently. Instead of adjusting all the parameters of a deep neural network, LoRA focuses on updating only a small set of low-rank matrices. 

By reducing the computational load, LoRA allows for the adaptation of these models on a much smaller scale of computing resources, making the use of large models more attainable for researchers and organizations with limited resources.

The focus of this Studio is to explain how LoRA works by coding it from scratch, which is an excellent exercise for looking under the hood of an algorithm.

You can read more about the LoRA from scratch implementation in this Studio's About page at https://lightning.ai/lightning-ai/studios/lora-from-scratch?view=public&section=all.


## Code

The code in this Studio is organized as follows:

### `00_lora-layer.ipynb` 

The `LoRALayer` and `LinearWithLoRA` standalone codes in a Jupyter Notebook if you want to toy around with it.

### `01_finetune-last-layers.ipynb` 

This Jupyter notebook implements a baseline DistilBERT model (a "small" LLM) where only the last few layers are finetuned to the target text classification dataset. You can run it from top to bottom to reproduce the results.



### `02_finetune-with-lora.ipynb` 

This Jupyter notebook augments the above-mentioned DistilBERT model using LoRA layers. You can run it from top to bottom to reproduce the results.



### `03_finetune-lora-script.py` 

This file is a more compact Python script implementation of the `02_finetune-with-lora.ipynb` notebook that can receive hyperparameter settings from the command line. 

For example, yo can run it from the VSCode terminal or standalone terminal as follows:

```bash
python 03_finetune-lora.py --lora_alpha 32 --lora_r 16
```


### `03_gridsearch.py` 

This file is a wrapper around `03_finetune-lora.py` executing a hyperparameter search. It saves the results to a `results.txt` file. 

You can either run it from the command line as `python 03_gridsearch.py` or submit it as a Job, as described in this Studio's About page (https://lightning.ai/lightning-ai/studios/lora-from-scratch?view=public&section=all). In either case, it's recommended to ensure you are on a machine with 4 GPUs when running this script.



### `04_finetune-all-layers.ipynb` 

This Jupyter notebook is a baseline similar to `01_finetune-last-layers.ipynb`. Whereas we only trained the last layer in `01_finetune-last-layers.ipynb`, this notebook trains all layers in the DistilBERT model.



### `local_dataset_utilities.py` and `local_model_utilities.py`

These are files containing utility functions for the dataset loading and model training to abstract away some commonly used code. The other files above import code from these files, but you don't need to use these files explicitely.# lora-from-scratch
