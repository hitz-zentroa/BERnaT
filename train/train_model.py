from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM, DataCollatorForLanguageModeling, HfArgumentParser, AutoConfig, set_seed
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk
from typing import Tuple
import torch
import wandb
import logging
from dataclasses import dataclass, field
from time import time
from accelerate import Accelerator
from accelerate.logging import get_logger
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RunArguments():

    model_name: str = field(
        default="bert-base-uncased",
        metadata={"help": "Model name"}
    )

    max_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the input"}
    )

    reference_tokenizer_name: str = field(
        default=None,
        metadata={"help": "Reference tokenizer name or path"}
    )

    attn_implementation: str = field(
        default=None,
        metadata={"help": "Attention implementation to use"}
    )

    train_data: str = field(
        default="Latxa",
        metadata={"help": "Training data"},
    )

    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Resume from checkpoint"}
    )

    from_scratch: bool = field(
        default=True,
        metadata={"help": "Train from scratch"}
    )

def load_model(model_name: str, from_scratch = True, **kwargs) -> AutoModelForMaskedLM:

    logger.info("Loading model...")

    if not from_scratch:
        model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
    else:
        config = AutoConfig.from_pretrained(model_name)

        model = AutoModelForMaskedLM.from_config(config, **kwargs)

    return model

def load_corpora(train_data) -> Tuple[Dataset, Dataset, Dataset]:

    logger.info("Loading corpora...")

    if train_data == "Standard":

        # Load the corpora
        egunkaria = load_dataset("HiTZ/latxa-corpus-v1.1", "egunkaria")
        booktegia = load_dataset("HiTZ/latxa-corpus-v1.1", "booktegi")
        oscar = load_dataset("HiTZ/latxa-corpus-v1.1", "colossal-oscar")
        culturax = load_dataset("HiTZ/latxa-corpus-v1.1", "culturax")
        euscrawl_1_1 = load_dataset("HiTZ/latxa-corpus-v1.1", "euscrawl-v1.1")
        hplt = load_dataset("HiTZ/latxa-corpus-v1.1", "hplt-v1")
        wikipedia = load_dataset("HiTZ/latxa-corpus-v1.1", "wikipedia")

        # Combine the corpora

        train_dataset: Dataset = concatenate_datasets([egunkaria['train'], booktegia['train'], oscar['train'], culturax['train'], euscrawl_1_1['train'], hplt['train'], wikipedia['train']])
        validation_dataset: Dataset = concatenate_datasets([egunkaria['validation'], booktegia['validation'], oscar['validation'], culturax['validation'], euscrawl_1_1['validation'], hplt['validation'], wikipedia['validation']])
        test_dataset: DatasetDict = DatasetDict({"egunkaria": egunkaria['test'], "booktegia": booktegia['test'], "oscar": oscar['test'], "culturax": culturax['test'], "euscrawl": euscrawl_1_1['test'], "hplt": hplt['test'], "wikipedia": wikipedia['test']})

    elif train_data == "Diverse":

        ekc = load_dataset("HiTZ/BERnaT-Diverse", "EKC")
        bsmauthor = load_dataset("HiTZ/BERnaT-Diverse", "BSMauthor")
        bsmtime = load_dataset("HiTZ/BERnaT-Diverse", "BSMtime")

        train_dataset: Dataset = concatenate_datasets([ekc['train'], bsmauthor['train'], bsmtime['train']])
        validation_dataset: Dataset = concatenate_datasets([ekc['validation'], bsmauthor['validation'], bsmtime['validation']])
        test_dataset: DatasetDict = DatasetDict({"EKC": ekc['test'], "BSMauthor": bsmauthor['test'], "BSMtime": bsmtime['test']})

    elif train_data == "Full":

        egunkaria = load_dataset("HiTZ/latxa-corpus-v1.1", "egunkaria")
        booktegia = load_dataset("HiTZ/latxa-corpus-v1.1", "booktegi")
        oscar = load_dataset("HiTZ/latxa-corpus-v1.1", "colossal-oscar")
        culturax = load_dataset("HiTZ/latxa-corpus-v1.1", "culturax")
        euscrawl_1_1 = load_dataset("HiTZ/latxa-corpus-v1.1", "euscrawl-v1.1")
        hplt = load_dataset("HiTZ/latxa-corpus-v1.1", "hplt-v1")
        wikipedia = load_dataset("HiTZ/latxa-corpus-v1.1", "wikipedia")

        ekc = load_dataset("HiTZ/BERnaT-Diverse", "EKC")
        bsmauthor = load_dataset("HiTZ/BERnaT-Diverse", "BSMauthor")
        bsmtime = load_dataset("HiTZ/BERnaT-Diverse", "BSMtime")

        train_dataset: Dataset = concatenate_datasets([egunkaria['train'], booktegia['train'], oscar['train'], culturax['train'], euscrawl_1_1['train'], hplt['train'], wikipedia['train'], ekc['train'], bsmauthor['train'], bsmtime['train']])
        validation_dataset: Dataset = concatenate_datasets([egunkaria['validation'], booktegia['validation'], oscar['validation'], culturax['validation'], euscrawl_1_1['validation'], hplt['validation'], wikipedia['validation'], ekc['validation'], bsmauthor['validation'], bsmtime['validation']])
        test_dataset: DatasetDict = DatasetDict({"egunkaria": egunkaria['test'], "booktegia": booktegia['test'], "oscar": oscar['test'], "culturax": culturax['test'], "euscrawl": euscrawl_1_1['test'], "hplt": hplt['test'], "wikipedia": wikipedia['test'], "EKC": ekc['test'], "BSMauthor": bsmauthor['test'], "BSMtime": bsmtime['test']})

    else:
        raise Exception("train_data not recognized")

    return train_dataset, validation_dataset, test_dataset

def preprocess_full_sentences_cross_boundaries(examples, tokenizer: AutoTokenizer = None, max_length: int = 512):
    inputs = []
    current_input = [tokenizer.bos_token_id]
    current_length = 0

    for example in examples['text']:
        sentences = example.split('.')
        
        for sentence in sentences:
            tokenized_sentence = tokenizer(sentence, add_special_tokens=False)['input_ids']
            if current_length + len(tokenized_sentence) + 1 > max_length:  # +1 for separator token
                if len(current_input) > 511:
                    current_input = current_input[:511] + [tokenizer.eos_token_id]
                inputs.append(current_input)
                current_input = [tokenizer.bos_token_id]
                current_length = 0

            current_input.extend(tokenized_sentence)
            #current_input.append(tokenizer.sep_token_id)
            current_length += len(tokenized_sentence) + 1

        if current_input and current_length + 1 <= max_length:
            current_input.append(tokenizer.sep_token_id)
            current_length += 1

    if current_input:
        inputs.append(current_input[:511] + [tokenizer.eos_token_id])
    
    return {'input_ids': inputs}

def preprocess_function(examples, tokenizer: AutoTokenizer = None):
    
    return tokenizer(examples["text"], return_special_tokens_mask=True)

def group_texts(examples, block_size=512):

    # Concatenate all texts.

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result


def main(trainingArguments: TrainingArguments, runArguments: RunArguments):

    data_path = "data"
    tokenizer_path = "tokenizer"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)

    model = load_model(runArguments.model_name, from_scratch=runArguments.from_scratch, attn_implementation=runArguments.attn_implementation)

    if Accelerator().is_main_process:

        # Avoid re-preprocessing if already done
        if not os.path.exists(f"{data_path}/{runArguments.train_data}"):
            os.makedirs(f"{data_path}/{runArguments.train_data}")
            train_dataset, validation_dataset, test_dataset = load_corpora(runArguments.train_data)
            if not os.path.exists(f"{tokenizer_path}/{runArguments.train_data}"):
                tokenizer = AutoTokenizer.from_pretrained(runArguments.reference_tokenizer_name)                
                logger.info("Training tokenizer...")
                tokenizer = tokenizer.train_new_from_iterator(train_dataset['text'], vocab_size=50005, min_frequency=5, show_progress=True, clean_up_tokenization_spaces=False)
                tokenizer.save_pretrained(f"{tokenizer_path}/{runArguments.train_data}")
                logger.info("Tokenizer trained and saved.")

                # Calculate token/word ratio in test splits
                inner_tokenizer = tokenizer._tokenizer
                def token_word_ratio(subsplit):
                    ratio_list = []
                    for text in subsplit:
                        if (tok:=(len(inner_tokenizer.encode(text).ids) - 2)) == 0 or (nltk:=(len(text.split()))) == 0:
                            continue
                        ratio_list.append(tok/nltk)
                    return ratio_list
                
                logger.info("Calculating token/word ratio in test splits...")
                for key in test_dataset.keys():
                    logger.info(f"Token/word ratio in {key}: {sum(token_word_ratio(test_dataset[key]['text']))/len(test_dataset[key])}")
            else:
                tokenizer = AutoTokenizer.from_pretrained(f"tokenizer/{runArguments.train_data}")

            logger.info("Preprocessing datasets...")

            tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer}, num_proc=8)
            tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer}, num_proc=8)
            tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer}, num_proc=8)

            chunked_train_dataset = tokenized_train_dataset.map(group_texts, batched=True, num_proc=8)
            chunked_validation_dataset = tokenized_validation_dataset.map(group_texts, batched=True, num_proc=8)
            chunked_test_dataset = tokenized_test_dataset.map(group_texts, batched=True, num_proc=8)

            chunked_train_dataset.save_to_disk(f"{data_path}/{runArguments.train_data}/train.data")
            chunked_validation_dataset.save_to_disk(f"{data_path}/{runArguments.train_data}/validation.data")
            chunked_test_dataset.save_to_disk(f"{data_path}/{runArguments.train_data}/test.data")

    # Main process will only preprocess the input

    Accelerator().wait_for_everyone()  # Sync all the threads

    # Ensure that every thread has the exact same data

    chunked_train_dataset = load_from_disk(f"{data_path}/{runArguments.train_data}/train.data")
    chunked_validation_dataset = load_from_disk(f"{data_path}/{runArguments.train_data}/validation.data")
    chunked_test_dataset = load_from_disk(f"{data_path}/{runArguments.train_data}/test.data")

    chunked_train_dataset = chunked_train_dataset.shuffle(seed=trainingArguments.seed)
    chunked_validation_dataset = chunked_validation_dataset.shuffle(seed=trainingArguments.seed)
    chunked_test_dataset = chunked_test_dataset.shuffle(seed=trainingArguments.seed)

    tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_path}/{runArguments.train_data}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=trainingArguments,
        train_dataset=chunked_train_dataset,
        eval_dataset=chunked_validation_dataset,
        data_collator=data_collator,
    )

    logger.info("Training...")

    st = time()

    trainer.train(resume_from_checkpoint=runArguments.resume_from_checkpoint)

    et = time()

    logger.info(f"Training took {et - st} seconds")

    if Accelerator().is_main_process:
        trainer.save_model(trainingArguments.output_dir)

    for key in chunked_test_dataset.keys():
        logger.info(f"Evaluating {key}...")
        logger.info(trainer.evaluate(chunked_test_dataset[key], metric_key_prefix="test"))

if __name__ == "__main__":

    yaml_path = sys.argv[1]
    parser = HfArgumentParser((RunArguments, TrainingArguments))
    runArguments, trainingArguments = parser.parse_yaml_file(yaml_file=yaml_path)

    set_seed(trainingArguments.seed)

    if Accelerator().is_main_process and trainingArguments.report_to == "wandb":
        run = wandb.init(project="BERnaT", name = trainingArguments.run_name)

    main(trainingArguments, runArguments)