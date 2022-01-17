from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, \
    EarlyStoppingCallback, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import argparse
import numpy as np
import os
from konlpy.tag import Kkma


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--language_model', type=str, default='KETI-AIR/ke-t5-base', help="")
parser.add_argument('--max_src_len', type=int, default=256, help="")
parser.add_argument('--max_tar_len', type=int, default=256, help="")
parser.add_argument('--truncation', type=boolean_string, default=True, help="")
parser.add_argument('--padding', type=str, default='max_length', help="")
parser.add_argument('--output_dir', type=str, default='runs/',
                    help="The output directory saves the models and configuration(default: 'runs/').")
parser.add_argument('--resume', type=str, default=None, help="The initial checkpoint for training(default: None).")
parser.add_argument('--do_train', type=boolean_string, default=True, help="Whether do training or not.")
parser.add_argument('--num_train_epochs', type=float, default=3.0, help="The number of train epochs(default:3.)")
parser.add_argument('--train_batch', type=int, default=4, help="The batch size of the train inputs(defualt:4).")
parser.add_argument('--max_train_sample', type=int, default=None,
                    help="The number of samples of train inputs(default:None, means no sampling).")
parser.add_argument("--max_steps", type=int, default=-1,
                    help="The number of steps while training(default:-1, means all steps).")
parser.add_argument("--logging_strategy", type=str, default="steps", help="The strategy for the log(default: epoch).")
parser.add_argument("--logging_steps", type=int, default=1000,
                    help="The interval number of log display(default: 1000).")
parser.add_argument("--save_steps", type=int, default=1000, help="The number of steps for saving model(default: 1000).")
parser.add_argument("--keep_ckpt_across_tasks", type=boolean_string, default=True,
                    help="Whether share the checkpoint across the tasks or not(default: True).")

parser.add_argument('--do_early_stopping', type=boolean_string, default=True,
                    help="Whether do the evaluation or not(default: False).")
parser.add_argument('--early_stopping_patience', type=int, default=3,
                    help="The number of tolerance for early stopping.")
parser.add_argument('--early_stopping_threshold', type=float, default=0.0,
                    help="The value of thresholds for early stopping.")

parser.add_argument('--do_eval', type=boolean_string, default=True,
                    help="Whether do the evaluation or not(default: True).")
parser.add_argument('--eval_split', type=boolean_string, default=False,
                    help="Whether split the train dataset for evaluation or not(default: False)")
parser.add_argument('--eval_batch', type=int, default=4, help="The number of batch size for the evaluation(de).")
parser.add_argument('--eval_steps', type=int, default=1000, help="The period step of the evaluation.")
parser.add_argument('--max_eval_sample', type=int, default=-1, help="The number of evaluation samples")
parser.add_argument("--eval_size", type=float, default=.1,
                    help="The split ratio of the training dataset for the evaluation(it is valid only when the 'eval_split' is True)")
parser.add_argument("--evaluation_strategy", type=str, default="steps",
                    help="The strategy for the evaluation(it should correspond the other strategy)")
parser.add_argument("--save_strategy", type=str, default="steps",
                    help="The strategy for saving the model(it should correspond the other strategy).")
parser.add_argument("--save_total_limit", type=int, default=None,
                    help="The maximum number of checkpoint files(If the number of checkpoint files is over the limit, the oldest checkpoint will be deleted automatically).")
parser.add_argument("--load_best_model_at_end", type=boolean_string, default=True, help="")

parser.add_argument('--do_test', type=boolean_string, default=False, help="Whether do the test or not(default: True).")
parser.add_argument('--path_prediction', type=str, default="test_output.json",
                    help="Prediction outputs for the test data.")
parser.add_argument('--metric_for_best_model', type=str, default=None, help="")
parser.add_argument('--generation_num_beams', type=int, default=1, help="")
parser.add_argument('--tpu_num_cores', type=int, default=None, help="")
parser.add_argument('--ignore_pad_token_for_loss', type=boolean_string, default=True, help='')
parser.add_argument('--source_prefix', type=str, default=None, help="")


def get_paraphrase_dataset(dataset, tokenizer, max_src_len=256, max_tar_len=256, truncation=True, padding='max_length'):
    def example_fn(examples):

        output = tokenizer(examples['sentence1'], truncation=truncation, padding=padding,
                           max_length=max_src_len)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['sentence2'], padding=padding, max_length=max_tar_len, truncation=truncation)

        output["labels"] = labels["input_ids"]
        output["is_paraphrased"] = [l['binary-label'] for l in examples['labels']]
        return output

    dataset = dataset.map(example_fn, batched=True)
    dataset = dataset.filter(lambda e: e["is_paraphrased"] == 1)
    return dataset


def get_tokenizer(language_model):
    if language_model.startswith('KETI-AIR/ke-t5'):
        return T5Tokenizer.from_pretrained(language_model)
    elif language_model=='koT5':
        return T5Tokenizer.from_pretrained(language_model)


def main():
    args = parser.parse_args()
    spliter = Kkma()
    dataset = load_dataset('klue', 'sts')
    tokenizer = get_tokenizer(args.language_model)
    train_dataset = get_paraphrase_dataset(dataset['train'], tokenizer, max_src_len=args.max_src_len,
                                           max_tar_len=args.max_tar_len, truncation=args.truncation,
                                           padding=args.padding)

    eval_dataset = get_paraphrase_dataset(dataset['validation'], tokenizer, max_src_len=args.max_src_len,
                                          max_tar_len=args.max_tar_len, truncation=args.truncation,
                                          padding=args.padding)
    
    if args.resume:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.resume)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.language_model)

    greater_is_better = args.metric_for_best_model if args.metric_for_best_model else "loss"
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, args.language_model),
        resume_from_checkpoint=args.resume,
        tpu_num_cores=args.tpu_num_cores,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_eval_batch_size=args.eval_batch,
        evaluation_strategy=args.evaluation_strategy if args.do_eval else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy if args.do_eval else "no",
        save_total_limit=args.save_total_limit,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=False if greater_is_better.find("loss") != -1 else True,
        load_best_model_at_end=args.load_best_model_at_end,
        label_smoothing_factor=0.0,
        sortish_sampler=False,
        generation_max_length=args.max_tar_len,
        generation_num_beams=args.generation_num_beams,
    )
    early_stop = None
    if args.do_early_stopping:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(spliter.sentences(pred)) for pred in preds]
        labels = ["\n".join(spliter.sentences(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        callbacks=[early_stop] if early_stop else None,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model('./koT5/')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    args = parser.