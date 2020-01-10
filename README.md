# oLMpics

This work was performed at The Allen Institute of Artificial Intelligence.

This project is constantly being improved. Contributions, comments and suggestions are welcome!

This repository contains the code for our paper [oLMpics - On what Language Model Pre-training Captures](https://https://arxiv.org/abs/1912.13283).

## Datasets

| Probe | Data   
| :----- | :-----:|  
| Always-Never | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/coffee_cats_quantifiers/coffee_cats_quantifiers_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/coffee_cats_quantifiers/coffee_cats_quantifiers_dev.jsonl.gz) 
| Age-Comparison | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/number_comparison/number_comparison_age_compare_masked_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/number_comparison/number_comparison_age_compare_masked_dev.jsonl.gz)  
| Objects-Comparison | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/size_comparison/size_comparison_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/size_comparison/size_comparison_dev.jsonl.gz) 
| Antonym Negation | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/antonym_synonym_negation/antonym_synonym_negation_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/antonym_synonym_negation/antonym_synonym_negation_dev.jsonl.gz)
| Taxonomy Conjunction  | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/hypernym_conjunction/hypernym_conjunction_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/hypernym_conjunction/hypernym_conjunction_dev.jsonl.gz)
| Encyclopedic Composition | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/composition/composition_v2_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/composition/composition_v2_dev.jsonl.gz)
| Multi-Hop Composition | [train](https://olmpics.s3.us-east-2.amazonaws.com/challenge/compositional_comparison/compositional_comparison_train.jsonl.gz) , [dev](https://olmpics.s3.us-east-2.amazonaws.com/challenge/compositional_comparison/compositional_comparison_dev.jsonl.gz)


## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/alontalmor/oLMpics.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd oLMpics
    ```

3.  Create a virtual environment with Python 3.6 or above:

    ```
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n olmpics python=3.7)
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use oLMpics.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh or conda activate olmpics)
    ```
5.  Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Multi-choice Masked Language Model (MC-MLM) training 

The AllenNLP train command is used for training. 
bert-base-uncased contains a simple test for a multi-choice language modeling baseline task (should currently
achieve an accuracy of ~0.99% at the last epoch). 

 `python -m allennlp.run train allennlp_models/config/transformer_masked_lm.jsonnet -s ../models_cache/roberta_local -o "{'dataset_reader': {'num_choices': '3', 'sample': '250','pretrained_model':'bert-base-uncased'}, 'validation_dataset_reader': {'num_choices': '3', 'sample': '-1', 'pretrained_model':'bert-base-uncased'}, 'iterator': {'batch_size': 8}, 'random_seed': '3', 'train_data_path': 's3://olmpics/challenge/multi_choice_language_modeling/multi_choice_language_modeling_train.jsonl.gz', 'model':{'pretrained_model': 'bert-base-uncased'}, 'trainer': {'cuda_device': -1, 'num_gradient_accumulation_steps': 2, 'learning_rate_scheduler': {'num_steps_per_epoch': 15}, 'num_epochs': '4'}, 'validation_data_path': 's3://olmpics/challenge/multi_choice_language_modeling/multi_choice_language_modeling_dev.jsonl.gz'}" --include-package allennlp_models`
 
* `'pretrained_model':'bert-base-uncased'`: one of the following LMs: bert-base-uncased, bert-large-uncased-whole-word-masking, bert-large-uncased, roberta-base, roberta-large
* `'sample': '250'`: number of training examples to sample. (To produce a learning curve)
* `'num_choices': '3'`: number of answer choices, depending on the task.  
* `random_seed`: an integer, we use (1,2,3,4,5,6) 

see allennlp_models/config/transformer_masked_lm.jsonnet for other options.

## Training the multi-choice question answering setup (MC-QA)

`python -m allennlp.run train allennlp_models/config/transformer_mc_qa.jsonnet -s ../models_cache/roberta_local -o "{'dataset_reader': {'num_choices': '3', 'sample': '100','pretrained_model':'bert-base-uncased'}, 'validation_dataset_reader': {'num_choices': '3', 'sample': '-1', 'pretrained_model':'bert-base-uncased'}, 'iterator': {'batch_size': 8}, 'random_seed': '3', 'train_data_path': 's3://olmpics/challenge/multi_choice_language_modeling/multi_choice_language_modeling_train.jsonl.gz', 'model':{'pretrained_model': 'bert-base-uncased'}, 'trainer': {'cuda_device': -1, 'num_gradient_accumulation_steps': 2, 'learning_rate_scheduler': {'num_steps_per_epoch': 15}, 'num_epochs': '4'}, 'validation_data_path': 's3://olmpics/challenge/multi_choice_language_modeling/multi_choice_language_modeling_dev.jsonl.gz'}" --include-package allennlp_models` 


## Training the MLM-Baseline

`python -m allennlp.run train train allennlp_models/config/mlm_baseline.jsonnet -s models/esim_local -o "{'dataset_reader':{'sample': '2000'}, 'iterator': {'batch_size': '16'}, 'random_seed': '2', 'train_data_path': 'https://olmpics.s3.us-east-2.amazonaws.com/challenge/same_item_or_not/same_item_or_not_train.jsonl.gz', 'trainer': {'cuda_device': -1, 'num_epochs': '60', 'num_serialized_models_to_keep': 0, 'optimizer': {'lr': 0.0004}}, 'validation_data_path': 'https://olmpics.s3.us-east-2.amazonaws.com/challenge/same_item_or_not/same_item_or_not_dev.jsonl.gz'}" --include-package allennlp_models` 

## Training the ESIM-Baseline
The esim baseline can be run as follows:

`python -m allennlp.run train train allennlp_models/config/esim_baseline.jsonnet -s models/esim_local -o "{'dataset_reader':{'sample': '500'}, 'iterator': {'batch_size': '16'}, 'random_seed': '2', 'train_data_path': 'https://olmpics.s3.us-east-2.amazonaws.com/challenge/multi_choice_language_modeling/multi_choice_language_modeling_train.jsonl.gz', 'trainer': {'cuda_device': -1, 'num_epochs': '60', 'num_serialized_models_to_keep': 0, 'optimizer': {'lr': 0.0004}}, 'validation_data_path': 'https://olmpics.s3.us-east-2.amazonaws.com/challenge/multi_choice_language_modeling/multi_choice_language_modeling_dev.jsonl.gz'}" --include-package allennlp_models` 

## Other
A caching infra is used, so make sure to have enough disk space, and control the cache directory using `OLMPICS_CACHE_ROOT` env variable.
see olmpics/common/file_utils.py





