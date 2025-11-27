from transformers import Trainer, TrainingArguments, set_seed, HfArgumentParser
import logging
import sys
import wandb
from dataclasses import dataclass, field
from time import time
from accelerate import PartialState
from accelerate.logging import get_logger
import torch
from sklearn.metrics import f1_score, accuracy_score
from evaluate import load as load_metric
from tasks_definitions import get_task
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@dataclass
class RunArguments():

    task: str = field(
        metadata={"help": "Task to train"}
    )

    model_name: str = field(
        default="bert-base-uncased",
        metadata={"help": "Model name"}
    )

    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Tokenizer name or path"}
    )

    attn_implementation: str = field(
        default=None,
        metadata={"help": "Attention implementation to use"}
    )

    from_scratch: bool = field(
        default=False,
        metadata={"help": "Train from scratch"}
    )

    max_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the input"}
    )

    percentage: float = field(
        default=1.0,
        metadata={"help": "Percentage of the training dataset to use"}
    )

def F1_eval(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    f1_value = f1_score(labels.flatten(), predictions.flatten(), average='micro')
    return {"F1": f1_value}

def ACC_eval(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_value = accuracy_score(labels, predictions)
    return {"ACC": acc_value}


f1_metric = load_metric("f1")

def MF1_eval(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_class = f1_metric.compute(references=labels.flatten(), predictions=predictions.flatten(), average=None)["f1"]
    f1_value =  (f1_class[1]+f1_class[2])/2
    return {"MF1": f1_value}

def SEQEVAL_eval(eval_preds_token):
    global label_names
    predictions_token, labels_token = eval_preds_token
    predictions_token = np.argmax(predictions_token, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels_token = [[label_names[l] for l in label if l != -100] for label in labels_token]
    true_predictions_token = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_token, labels_token)
    ]
    all_metrics_token = seqeval.compute(predictions=true_predictions_token, references=true_labels_token)
    metrics =  {
        "SEQEVAL": all_metrics_token["overall_f1"],
    }
    return metrics

seqeval = None
label_names = None

def main(trainingArguments: TrainingArguments, runArguments: RunArguments):
    global label_names, seqeval
    experiment_id = f'{trainingArguments.run_name}_{runArguments.task}_{trainingArguments.seed}'
    seqeval = load_metric("seqeval", experiment_id=experiment_id)

    set_seed(trainingArguments.seed)
    logger.info(f"Evaluating {runArguments.task} task...")

    task = get_task(runArguments.task, from_scratch=runArguments.from_scratch)
    model, tokenizer, train_dataset, validation_dataset, test_dataset, collator = task.get_model_tokenizer_dataset_collator(runArguments.model_name, runArguments.tokenizer_name, runArguments.max_length, percentage=runArguments.percentage, attn_implementation=runArguments.attn_implementation, torch_dtype=torch.float16)
    
    if runArguments.percentage != 1.0:
        num_train_steps = (task._orig_size // (trainingArguments.per_device_train_batch_size * trainingArguments.gradient_accumulation_steps)) * trainingArguments.num_train_epochs
        trainingArguments.max_steps = num_train_steps
        trainingArguments.num_train_epochs = 0
        logging.info(f"Training for {num_train_steps} steps")
        logging.info(f"Training for {len(train_dataset)} examples, originally {task._orig_size} examples")
    else:
        logging.info(f"Training for {trainingArguments.num_train_epochs} epochs")
        logging.info(f"Training for {len(train_dataset)} examples")

    label_names = task.id2label
    metric_function = globals()[f"{task.metric}_eval"]
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=trainingArguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics= metric_function,
    )

    logger.info("Training...")
    st = time()
    trainer.train()
    et = time()
    logger.info(f"Training took {et - st} seconds")

    for test_name, test_dataset in test_dataset:
        logger.info(f"Evaluating {test_name}...")
        results = trainer.evaluate(test_dataset, metric_key_prefix=test_name)
        logger.info(results)

if __name__ == "__main__":

    yaml_path = sys.argv[1]

    parser = HfArgumentParser((RunArguments, TrainingArguments))
    runArguments, trainingArguments = parser.parse_yaml_file(yaml_file=yaml_path)

    if PartialState().is_main_process and trainingArguments.report_to == "wandb":
        run = wandb.init(project="BERnaT", name = trainingArguments.run_name)
        yaml_file = wandb.Artifact("yaml", type="yaml", description="Yaml file used for finetuning")
        run.log_artifact(yaml_file)

    main(trainingArguments, runArguments)