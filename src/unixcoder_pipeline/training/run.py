# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
from typing import Dict, Tuple, List

import hydra
import wandb

from unixcoder_pipeline.data_processing.process_features import read_examples, convert_examples_to_features, Example
from unixcoder_pipeline.training.metrics import bleu
from omegaconf import DictConfig
import torch
import random
import logging
import numpy as np
from io import open
from unixcoder_pipeline.model.model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)

from transformers import (  # type: ignore
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)

from unixcoder_pipeline.utils import set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="finetune", version_base=None)
def main(training_config: DictConfig):
    # set log
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config.n_gpu = torch.cuda.device_count()
    # training_config.device = device
    logger.info("device: %s, n_gpu: %s", device, training_config.n_gpu)

    # Set seed
    set_seed(training_config.seed)

    # make dir if output_dir not exist
    if os.path.exists(training_config.output_dir) is False:
        os.makedirs(training_config.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(training_config.model_name_or_path)
    config = RobertaConfig.from_pretrained(training_config.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(training_config.model_name_or_path, config=config)

    model = Seq2Seq(
        encoder=encoder,
        decoder=encoder,
        config=config,
        beam_size=training_config.beam_size,
        max_length=training_config.max_target_length,
        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
        eos_id=tokenizer.sep_token_id,
    )

    logger.info("Training/evaluation parameters %s", training_config)
    model.to(device)

    # if training_config.n_gpu > 1:
    #     # multi-gpu training
    #     model = torch.nn.DataParallel(model)
    wandb_config = {"architecture": "UnixCoder", "learning_rate": training_config.learning_rate,
                    "num_train_epochs": training_config.num_train_epochs,
                    "train_batch_size": training_config.train_batch_size, "beam_size": training_config.beam_size,
                    "max_source_length": training_config.max_source_length,
                    "max_target_length": training_config.max_target_length}
    if training_config.do_train:
        wandb.init(
            # set the wandb project where this run will be logged
            project="code-summarization",

            # track hyperparameters and run metadata
            config=wandb_config
        )
        wandb.define_metric("eval_bleu", summary="max")
        wandb.define_metric("eval_ppl", summary="min")

        # Prepare training data loader
        train_examples = read_examples(training_config.train_filename)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, training_config, logger, stage="train"
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in train_features], dtype=torch.long
        )
        all_target_ids = torch.tensor(
            [f.target_ids for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(all_source_ids, all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=training_config.train_batch_size // training_config.gradient_accumulation_steps,
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=training_config.learning_rate, eps=training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train_dataloader) * training_config.num_train_epochs * 0.1),
            num_training_steps=len(train_dataloader) * training_config.num_train_epochs,
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info(
            "  Batch size = %d",
            training_config.train_batch_size * training_config.gradient_accumulation_steps,
        )
        logger.info("  Num epoch = %d", training_config.num_train_epochs)

        model.train()
        patience, best_bleu, losses = 0, 0, []
        dev_dataset: Dict[str, Tuple[List[Example], TensorDataset]] = {}
        for epoch in range(training_config.num_train_epochs):
            for idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids = batch
                loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)

                if training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if training_config.gradient_accumulation_steps > 1:
                    loss = loss / training_config.gradient_accumulation_steps

                losses.append(loss.item())
                loss.backward()
                if len(losses) % training_config.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // training_config.gradient_accumulation_steps % 100 == 0:
                        loss_val = round(
                            float(np.mean(losses[-100 * training_config.gradient_accumulation_steps:])),
                            4,
                        )
                        wandb.log(
                            {"train-loss": loss_val}
                        )
                        logger.info(
                            "epoch {} step {} loss {}".format(
                                epoch,
                                len(losses) // training_config.gradient_accumulation_steps,
                                loss_val,
                            )
                        )
            if training_config.do_eval:
                # Eval model with dev dataset
                if "dev_loss" in dev_dataset:
                    eval_examples, eval_data = dev_dataset["dev_loss"]
                else:
                    eval_examples = read_examples(training_config.dev_filename)
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, training_config, logger, stage="dev"
                    )
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long
                    )
                    all_target_ids = torch.tensor(
                        [f.target_ids for f in eval_features], dtype=torch.long
                    )
                    eval_data = TensorDataset(all_source_ids, all_target_ids)
                    dev_dataset["dev_loss"] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=training_config.eval_batch_size
                )

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", training_config.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0., 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, target_ids = batch

                    with torch.no_grad():
                        _, loss, num = model(
                            source_ids=source_ids, target_ids=target_ids
                        )
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {"eval_ppl": round(np.exp(eval_loss), 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # Calculate bleu
                if "dev_bleu" in dev_dataset:
                    eval_examples, eval_data = dev_dataset["dev_bleu"]
                else:
                    eval_examples = read_examples(training_config.dev_filename)
                    eval_examples = random.sample(
                        eval_examples, min(1000, len(eval_examples))
                    )
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, training_config, logger, stage="test"
                    )
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long
                    )
                    eval_data = TensorDataset(all_source_ids)
                    dev_dataset["dev_bleu"] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=training_config.eval_batch_size
                )

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]
                    with torch.no_grad():
                        preds = model(source_ids)
                        # convert ids to text
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[: t.index(0)]
                            text = tokenizer.decode(
                                t, clean_up_tokenization_spaces=False
                            )
                            p.append(text)
                model.train()
                predictions = []
                with open(training_config.output_dir + "/dev.output", "w") as f, open(
                        training_config.output_dir + "/dev.gold", "w"
                ) as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + "\t" + ref)
                        f.write(str(gold.idx) + "\t" + ref + "\n")
                        f1.write(str(gold.idx) + "\t" + gold.target + "\n")

                (goldMap, predictionMap) = bleu.compute_maps(
                    predictions, os.path.join(training_config.output_dir, "dev.gold")
                )
                dev_bleu = round(bleu.bleu_from_maps(goldMap, predictionMap)[0], 2)
                wandb.log(
                    {
                        "eval_bleu": float(dev_bleu),
                        "eval_ppl": round(np.exp(eval_loss), 5)
                    }
                )
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(training_config.output_dir, "checkpoint-best-bleu")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience = 0
                else:
                    patience += 1
                    if patience == 2:
                        break
        wandb.finish()

    if training_config.do_test:
        checkpoint_prefix = "checkpoint-best-bleu/pytorch_model.bin"
        output_dir = os.path.join(training_config.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(torch.load(output_dir))

        eval_examples = read_examples(training_config.test_filename)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, training_config, logger, stage="test"
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(all_source_ids)

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=training_config.eval_batch_size
        )

        model.eval()
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            with torch.no_grad():
                preds = model(source_ids)
                # convert ids to text
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)

        model.train()
        predictions = []
        with open(training_config.output_dir + "/test.output", "w") as f, open(
                training_config.output_dir + "/test.gold", "w"
        ) as f1:
            for ref, gold in zip(p, eval_examples):
                predictions.append(str(gold.idx) + "\t" + ref)
                f.write(str(gold.idx) + "\t" + ref + "\n")
                f1.write(str(gold.idx) + "\t" + gold.target + "\n")

        (goldMap, predictionMap) = bleu.compute_maps(
            predictions, os.path.join(training_config.output_dir, "test.gold")
        )
        dev_bleu = round(bleu.bleu_from_maps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
