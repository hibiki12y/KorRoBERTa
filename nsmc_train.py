# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import math
import os
import json
import logging

from tqdm import tqdm
from utils import set_seed, init_gpu_params
from dataset import NSMCDataSet

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import get_linear_schedule_with_warmup, AdamW

from nsmc_modeling import RobertaForSequenceClassification
from bert.tokenizer import Tokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sanity_checks(params):
    if os.path.isdir(params.save_checkpoints_dir):
        assert not os.listdir(params.save_checkpoints_dir), "Result directory must be empty"
    else:
        os.mkdir(params.save_checkpoints_dir)


def _get_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("--bert_model", type=str, default='bert/')
    parser.add_argument("--pretrained_bert_model", type=str, default="output/senti_base_model_100000step")
    parser.add_argument("--device_ids", type=str, default="0")

    parser.add_argument("--seed", type=int, default=203)

    parser.add_argument("--save_checkpoints_dir", type=str, default="result/")
    parser.add_argument("--save_checkpoints_steps", type=int, default=100)

    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)

    parser.add_argument("--classifier_dropout", type=float, default=0.0)

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--pad_to_max", action="store_true")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")

    return parser


def main():
    args = _get_parser().parse_args()

    args.device_ids = list(map(int, args.device_ids.split(',')))

    set_seed(args)
    sanity_checks(args)
    init_gpu_params(args)

    tokenizer = Tokenizer(os.path.join(args.bert_model, "senti_vocab.txt"),
                          os.path.join(args.bert_model, "RoBERTa_Sentiment_kor"))

    train_dataset = NSMCDataSet(data_split="train",
                                tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length,
                                pad_to_max=args.pad_to_max)

    train_sampler = RandomSampler(train_dataset) if not args.multi_gpu else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.per_gpu_train_batch_size,
                                  collate_fn=train_dataset.collate_fn)

    model = RobertaForSequenceClassification(classifier_dropout=args.classifier_dropout,bert_model_dir=args.bert_model,pre_trained_model=args.pretrained_bert_model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    model.zero_grad()
    model.cuda()

    if args.multi_gpu:
        model = DistributedDataParallel(model,
                                        device_ids=[args.device_ids[args.local_rank]],
                                        output_device=args.device_ids[args.local_rank])

    if args.is_master:
        logger.info(json.dumps(vars(args), indent=4))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.per_gpu_train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.multi_gpu else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_steps = 0
    for epoch in range(args.num_train_epochs):
        if args.multi_gpu:
            train_sampler.set_epoch(epoch)

        loss_bce = nn.BCEWithLogitsLoss()

        iter_loss = 0

        model.train()

        pbar = tqdm(train_dataloader, desc="Iter", disable=not args.is_master)
        for step, batch in enumerate(pbar):
            input_ids, attention_mask, labels = batch

            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).cuda(),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).cuda()
            }
            logits = model(**inputs)

            labels = torch.tensor(labels, dtype=torch.float).cuda()

            loss = loss_bce(input=logits.view(-1), target=labels.view(-1))
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_steps += 1
                if global_steps % args.save_checkpoints_steps == 0 and args.is_master:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    save_path = os.path.join(args.save_checkpoints_dir, f"step_{global_steps}.ckpt")
                    torch.save(model_to_save.state_dict(), save_path)

            iter_loss += loss.item()
            pbar.set_postfix({
                "epoch": epoch,
                "global_steps": global_steps,
                "learning_rate": f"{scheduler.get_last_lr()[0]:.10f}",
                "avg_iter_loss": f"{iter_loss / (step + 1) * args.gradient_accumulation_steps:.5f}",
                "last_loss": f"{loss.item() * args.gradient_accumulation_steps:.5f}"})
        pbar.close()

        if args.is_master:
            model_to_save = model.module if hasattr(model, 'module') else model
            save_path = os.path.join(args.save_checkpoints_dir, f"epoch_{epoch+1}.ckpt")
            torch.save(model_to_save.state_dict(), save_path)


if __name__ == "__main__":
    main()
