import time
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

from code_summarization.src.run import Example, InputFeatures, read_examples, convert_examples_to_features
from code_summarization.src.model import Seq2Seq


def main():

    arg_config = {
        'model_name_or_path': "microsoft/unixcoder-base",
        'beam_size': 10,
        'max_target_length': 128,
        'max_source_length': 256,
        'eval_batch_size': 1
    }

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(arg_config['model_name_or_path'])
    config = RobertaConfig.from_pretrained(arg_config['model_name_or_path'])
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(arg_config['model_name_or_path'], config=config)

    model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                    beam_size=arg_config['beam_size'], max_length=arg_config['max_target_length'],
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)

    test_filename = "../dataset/python/test.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    eval_examples = read_examples(test_filename)

    generation_lengths = namedtuple("GenerationLengths", ["max_source_length", "max_target_length"])
    generation_lengths.max_target_length = arg_config['max_target_length']
    generation_lengths.max_source_length = arg_config['max_source_length']

    eval_features = convert_examples_to_features(eval_examples, tokenizer, generation_lengths, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids)

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=arg_config['eval_batch_size'])

    model.eval()
    p = []
    times = []
    n_examples = 10
    for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids = batch[0]
        with torch.no_grad():
            start_time = time.time()
            preds = model(source_ids)
            end_time = time.time()
            times.append(end_time - start_time)
            # convert ids to text
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
        if i == n_examples - 1:
            break
    print(f"{np.mean(times)} for {n_examples}")


if __name__ == "__main__":
    main()