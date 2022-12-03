import json
import os
import logging
import pandas as pd
import argparse
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataset(path_to_data):
	"""
	Dataset should be a list of lists of text, label pairs.
	"""
	with open(path_to_data) as f:
		data = json.loads(f.read())
	data = [[x['description'], 1 if x['label']=='bot' else 0] for x in data]
	df = pd.DataFrame(data)
	df.columns = ["text", "labels"]
	return df


def make_model(args):

	model_args = ClassificationArgs(num_train_epochs=args.num_train_epochs, 
                                    do_lower_case=args.do_lower_case,
                                    early_stopping_consider_epochs=args.early_stopping_consider_epochs,
                                    train_batch_size=args.train_batch_size,
                                    eval_batch_size=args.eval_batch_size,
                               	    evaluate_during_training=args.evaluate_during_training,
                                    learning_rate=args.learning_rate,
                                    output_dir=args.output_dir,
                                    overwrite_output_dir=args.overwrite_output_dir,
                                    manual_seed=args.manual_seed,
                                    use_early_stopping=args.use_early_stopping,
                                    early_stopping_metric=args.early_stopping_metric,
                                    sliding_window=args.sliding_window,
                                    stride=args.stride
                                    )

	model = ClassificationModel(
	    args.model_type, args.tokenizer_name, 
	    args=model_args, use_cuda=args.use_cuda
	)
	return model


def get_metrics(result):
    
    accuracy = (result['tp'] + result['tn']) / (result['tp'] + result['tn'] + result['fp'] + result['fn'])
    precision = result['tp'] / (result['tp'] + result['fp'])
    recall = result['tp'] / (result['tp'] + result['fn'])
    
    return {'acc' : accuracy, 'precision':precision, 'recall' : recall}


def train_test(args):

	logging.basicConfig(level=logging.ERROR)
	transformers_logger = logging.getLogger("transformers")
	transformers_logger.setLevel(logging.WARNING)

	train_data = get_dataset(args.path_to_train)
	val_data = get_dataset(args.path_to_val)
	test_data = get_dataset(args.path_to_test)

	model = make_model(args)
	model.train_model(train_data, eval_df=val_data)

	result, model_outputs, wrong_predictions = model.eval_model(test_data)
	metrics = get_metrics(result)

	print("Test results = ", metrics)

	return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_train", type=str)
    parser.add_argument("--path_to_val", type=str)
    parser.add_argument("--path_to_test", type=str)
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--model_type", type=str, default='roberta')
    parser.add_argument("--tokenizer_name", type=str, default='roberta-base')
    
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--do_lower_case", type=bool, default=True)
    parser.add_argument("--early_stopping_consider_epochs", type=bool, default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--evaluate_during_training", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--output_dir", type=str, default='./outputs/output')
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--use_early_stopping", type=bool, default=True)
    parser.add_argument("--early_stopping_metric", type=str, default='eval_loss')
    parser.add_argument("--sliding_window", type=bool, default=False)
    parser.add_argument("--stride", type=float, default=0.8)
    parser.add_argument("--manual_seed", type=int, default=11711)
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")

    print(args)
    return args


if __name__ == "__main__":

	args = get_args()
	train_test(args)







