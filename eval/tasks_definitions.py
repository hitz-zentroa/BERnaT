from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding, AutoConfig
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

BASQUE_GLUE_TASKS = {
    'bec': ["text-classification", "F1"],
    'bhtc': ["text-classification", "F1"],
    'coref': ["coreference-resolution", "ACC"],
    'intent': ["text-classification", "F1"],
    'nerc_id': ["token-classification", "SEQEVAL"],
    'nerc_od': ["token-classification", "SEQEVAL"],
    'nerc': ["token-classification-all_in", "SEQEVAL"],
    'qnli': ["qnli", "ACC"],
    'slot': ["token-classification", "SEQEVAL"],
    'vaxx': ["text-classification", "MF1"],
    'wic': ["wic", "ACC"],
    'pos': ["pos", "SEQEVAL"],
    'pos_h_orig': ["pos_h_orig", "SEQEVAL"],
    'pos_h_norm': ["pos_h_norm", "SEQEVAL"],
}


########## POS UTILS
id_to_tag = {
    0: "ADJ",
    1: "ADP",
    2: "ADV",
    3: "AUX",
    4: "CCONJ",
    5: "DET",
    6: "INTJ",
    7: "NOUN",
    8: "NUM",
    9: "PART",
    10: "PRON",
    11: "PROPN",
    12: "PUNCT",
    13: "SCONJ",
    14: "SYM",
    15: "VERB",
    16: "X"
}

def patch_data(data):
    # Convert from a list of dict to a dict of list

    data2 = {}
    for split in data.keys():
        data2[split] = {}
        for i in range(len(data[split])):
            for key in data[split][i].keys():
                if key not in data2[split]:
                    data2[split][key] = []
                data2[split][key].append(data[split][i][key])
    return data2
####################

class TASK:

    def __init__(self, name, task_type, metric, from_scratch = False):
        self.name = name
        self.task_type = task_type
        self.metric = metric
        self.from_scratch = from_scratch

    def _create_model(self, model_name, torch_dtype=None, attn_implementation=None):

        config = AutoConfig.from_pretrained(model_name)

        if "bert" in config.model_type.lower(): # Issues with torch_dtype and bert mode
            torch_dtype = None
        if not self.from_scratch:
            return self.model_type.from_pretrained(model_name, num_labels=self.num_labels, label2id=self.label2id, id2label=self.id2label, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        else:
            model = self.model_type.from_config(config)
            model.resize_token_embeddings(len(self.tokenizer))
            return model
    
    def _load_tokenizer(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def _load_corpora(self, percentage = 1.0):

        dataset = load_dataset("orai-nlp/basqueGLUE", self.name)

        self._orig_size = len(dataset["train"])

        if percentage != 1.0:
            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * percentage)))
        
        self.train_dataset = dataset['train']
        self.validation_dataset = dataset['validation']
        self.test_dataset = [("test", dataset['test'])]

    def _calculate_labels(self):

        self.labels = self.train_dataset.features["label"].names

        self.num_labels = len(self.labels)

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def _tokenize_dataset(self, dataset: Dataset, max_length: int):
        raise NotImplementedError
    
    def _create_collator(self):
        raise NotImplementedError

    def get_model_tokenizer_dataset_collator(self, model_name, tokenizer_name, max_length, percentage = 1.0, **kwargs):

        if tokenizer_name is None:    
            tokenizer_name = model_name
            
        self._load_tokenizer(tokenizer_name)

        self._load_corpora(percentage=percentage)

        self._calculate_labels()

        self.model = self._create_model(model_name, **kwargs)

        self.train_dataset = self._tokenize_dataset(self.train_dataset, max_length)
        self.validation_dataset = self._tokenize_dataset(self.validation_dataset, max_length)
        tokenized_tests = []
        for name_test in self.test_dataset:
            name, test = name_test
            test_dataset = self._tokenize_dataset(test, max_length)
            tokenized_tests.append((name, test_dataset))

        self.test_dataset = tokenized_tests

        self._create_collator(max_length)

        return self.model, self.tokenizer, self.train_dataset, self.validation_dataset, self.test_dataset, self.collator

class TEXT_CLASSIFICATION_TASK(TASK):

    def __init__(self, name, metric, from_scratch = False):
        super().__init__(name, "text-classification", metric, from_scratch)
        self.model_type = AutoModelForSequenceClassification

    def _tokenize_dataset(self, dataset: Dataset, max_length: int):

        return dataset.map(lambda examples: self.tokenizer(examples["text"], truncation=True, max_length=max_length), batched=True, num_proc=4)

    def _create_collator(self, max_length):

        self.collator = DataCollatorWithPadding(self.tokenizer, max_length=max_length)


class TOKEN_CLASSIFICATION_TASK(TASK):

    def __init__(self, name, metric, from_scratch = False):
        super().__init__(name, "token-classification", metric, from_scratch)
        self.model_type = AutoModelForTokenClassification
    
    def _load_tokenizer(self, tokenizer_name: str):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, add_prefix_space=True)

    def _calculate_labels(self):

        if self.name == "nerc_id" or self.name == "nerc_od" or self.name == "nerc":    

            self.labels = ["O",
                "B-PER",
                "I-PER",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC"]
        
        elif self.name == "pos":

            self.labels = [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X"
            ]
        
        elif  self.name == "pos_h_orig" or self.name == "pos_h_norm":

            self.labels = [
                'ADJ',
                'ADV',
                'AUX',
                'CONJ',
                'DET',
                'INTJ',
                'MWE',
                'NOUN',
                'NUM',
                'PART',
                'PRON',
                'PROPN',
                'PUNCT',
                'VERB',
                'X'
            ]

        else:
            self.labels = self.train_dataset.features["tags"].feature.names

        self.num_labels = len(self.labels)

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def _tokenize_and_align_labels(self, examples, tokenizer: AutoTokenizer, max_length):

        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, max_length=max_length, is_split_into_words=True)

        labels = []

        for i, label in enumerate(examples["tags"]):

            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.

            previous_word_idx = None

            label_ids = []

            for word_idx in word_ids:  # Set the special tokens to -100.

                if word_idx is None:

                    label_ids.append(-100)

                elif word_idx != previous_word_idx:  # Only label the first token of a given word.

                    label_ids.append(label[word_idx])

                else:

                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def _tokenize_dataset(self, dataset: Dataset, max_length: int):

        return dataset.map(lambda examples: self._tokenize_and_align_labels(examples, self.tokenizer, max_length=max_length), batched=True, num_proc=4)
    
    def _create_collator(self, max_length):

        self.collator = DataCollatorForTokenClassification(self.tokenizer, padding=True, max_length=max_length)

class COMPOSED_INPUT_TASKS(TASK):

    def __init__(self, name, metric, from_scratch = False):
        super().__init__(name, "text-classification", metric, from_scratch)
        self.model_type = AutoModelForSequenceClassification
    
    def _create_instance(self, example):

        pass

    def _tokenize_dataset(self, dataset: Dataset, max_length: int):

        return  dataset.map(lambda examples: self.tokenizer(self._create_instance(examples), truncation=True, max_length=max_length), batched=True, num_proc=1)
    
    def _create_collator(self, max_length):

        self.collator = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=max_length)


class COREFERENCE_RESOLUTION_TASK(COMPOSED_INPUT_TASKS):

    def _create_instance(self, example):

        return list(map(lambda x, y, z: f'{x}</s></s>{y}</s></s>{z}', zip(example["text"]), example["span1_text"], example["span2_text"]))

class QNLI_TASK(COMPOSED_INPUT_TASKS):

    def _create_instance(self, example):
            
        return list(map(lambda x, y: f'{x}</s></s>{y}', example["question"], example["sentence"]))

class WIC_TASK(COMPOSED_INPUT_TASKS):

    def _create_instance(self, example):
            
        return list(map(lambda x, y, z: f'{x}</s></s>{y}</s></s>{z}', zip(example["sentence1"]), example["sentence2"], example["word"]))
    
    def _load_corpora(self, percentage = 1.0):
        super()._load_corpora(percentage=percentage)

        self.train_dataset = self.train_dataset.shuffle(seed=42).select(range(40_000))

class TOKEN_CLASSIFICATION_ALL_IN_TASK(TOKEN_CLASSIFICATION_TASK):

    def _load_corpora(self, percentage = 1.0):

        nerc_id = load_dataset("orai-nlp/basqueGLUE", "nerc_id")
        nerc_od = load_dataset("orai-nlp/basqueGLUE", "nerc_od")

        # Merge the two datasets
        self.train_dataset = Dataset.from_pandas(pd.concat([nerc_id["train"].to_pandas(), nerc_od["train"].to_pandas()]), preserve_index=False)
        self.validation_dataset = Dataset.from_pandas(pd.concat([nerc_id["validation"].to_pandas(), nerc_od["validation"].to_pandas()]), preserve_index=False)
        self.test_dataset = [("test" ,Dataset.from_pandas(pd.concat([nerc_id["test"].to_pandas(), nerc_od["test"].to_pandas()]), preserve_index=False))]

        self._orig_size = len(self.train_dataset)

        self.labels = list(set(nerc_id["train"].features["tags"].feature.names) | set(nerc_od["train"].features["tags"].feature.names))

class POS_TASK(TOKEN_CLASSIFICATION_TASK):

    def _load_corpora(self, percentage = 1.0):

        data = load_dataset("HiTZ/PoSud")

        self.train_dataset = Dataset.from_dict(data["train"])
        self.validation_dataset = Dataset.from_dict(data["validation"])
        self.test_dataset = [("test", Dataset.from_dict(data["test"]))]

        # Rename the column to "tags"
        self.train_dataset = self.train_dataset.rename_column("upos", "tags")
        self.validation_dataset = self.validation_dataset.rename_column("upos", "tags")
        self.test_dataset = [(name, ds.rename_column("upos", "tags")) for name, ds in self.test_dataset]

        self._orig_size = len(self.train_dataset)

        if percentage != 1.0:
            self.train_dataset = self.train_dataset.shuffle(seed=42).select(range(int(len(self.train_dataset) * percentage)))

class PoS_HISTORIKOA_TASK(TOKEN_CLASSIFICATION_TASK):

    def __init__(self, name, metric, from_scratch = False, train_type = "orig"):
        super().__init__(name, metric, from_scratch)
        self.train_type = train_type

    def _split_words_words_norm(self, dataset):
    
        # Word
        part1 = {
            "word": [],
            "udp": [],
            "book_name": [],
            "euskalkia": []
        }
        # Word_norm
        part2 = {
            "word": [],
            "udp": [],
            "book_name": [],
            "euskalkia": []
        }
        for index, row in dataset.to_pandas().iterrows():
            part1["word"].append(row["word"])
            part1["udp"].append(row["udp"])
            part1["book_name"].append(row["book_name"])
            part1["euskalkia"].append(row["euskalkia"])

            part2["word"].append(row["word_norm"])
            part2["udp"].append(row["udp"])
            part2["book_name"].append(row["book_name"])
            part2["euskalkia"].append(row["euskalkia"])

        part1_df = pd.DataFrame(part1)
        part2_df = pd.DataFrame(part2)
        return Dataset.from_pandas(part1_df), Dataset.from_pandas(part2_df)

    def _load_corpora(self, percentage = 1.0):

        dataset = load_dataset("HiTZ/PoShis")

        self.train_dataset = dataset['train']
        self.validation_dataset = dataset['validation']

        test_orig, test_norm = self._split_words_words_norm(dataset['test'])
        self.test_dataset = [("test_orig", test_orig), ("test_norm", test_norm)]

        # Rename the column to words to tokens and udp to tags
        if self.train_type == "orig":
            self.train_dataset = self.train_dataset.rename_column("word", "tokens").rename_column("udp", "tags")
            self.validation_dataset = self.validation_dataset.rename_column("word", "tokens").rename_column("udp", "tags")
        else:
            self.train_dataset = self.train_dataset.rename_column("word_norm", "tokens").rename_column("udp", "tags")
            self.validation_dataset = self.validation_dataset.rename_column("word_norm", "tokens").rename_column("udp", "tags")

        self.test_dataset = [(name, ds.rename_column("word", "tokens").rename_column("udp", "tags")) for name, ds in self.test_dataset]

        self._orig_size = len(self.train_dataset)

        if percentage != 1.0:
            self.train_dataset = self.train_dataset.shuffle(seed=42).select(range(int(len(self.train_dataset) * percentage)))

    def _map_labels_to_ids(self, examples):
        examples["tags"] = [self.label2id[tag] for tag in examples["tags"]]
        return examples

    def _calculate_labels(self):

        super()._calculate_labels()

        # Map labels to ids
        self.train_dataset = self.train_dataset.map(self._map_labels_to_ids, batched=False)
        self.validation_dataset = self.validation_dataset.map(self._map_labels_to_ids, batched=False)
        tokenized_tests = []
        for name_test in self.test_dataset:
            name, test = name_test
            test_dataset = test.map(self._map_labels_to_ids, batched=False)
            tokenized_tests.append((name, test_dataset))

        self.test_dataset = tokenized_tests

def get_task(name: str, from_scratch: bool = False) -> TASK:

    task_type, metric = BASQUE_GLUE_TASKS[name]

    if task_type == "text-classification":

        return TEXT_CLASSIFICATION_TASK(name, metric, from_scratch)

    elif task_type == "token-classification":

        return TOKEN_CLASSIFICATION_TASK(name, metric, from_scratch)
    
    elif task_type == "token-classification-all_in":

        return TOKEN_CLASSIFICATION_ALL_IN_TASK(name, metric, from_scratch)

    elif task_type == "qnli":

        return QNLI_TASK(name, metric, from_scratch)

    elif task_type == "coreference-resolution":

        return COREFERENCE_RESOLUTION_TASK(name, metric, from_scratch)
    
    elif task_type == "wic":

        return WIC_TASK(name, metric, from_scratch)

    elif task_type == "pos":

        return POS_TASK(name, metric, from_scratch)

    elif task_type == "pos_h_orig":

        return PoS_HISTORIKOA_TASK(name, metric, from_scratch, train_type="orig")

    elif task_type == "pos_h_norm":

        return PoS_HISTORIKOA_TASK(name, metric, from_scratch, train_type="norm")

    else:

        raise NotImplementedError(f"Task type {task_type} not implemented")
