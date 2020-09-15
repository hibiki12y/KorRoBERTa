import torch
import argparse
import os
import glob

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from nsmc_modeling import RobertaForSequenceClassification
from bert.tokenizer import Tokenizer
from dataset import NSMCDataSet


def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state_dict", type=str, required=True)
    parser.add_argument("--bert_model", type=str, default='bert/')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--no_display", action="store_true")

    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()

    tokenizer = Tokenizer(os.path.join(args.bert_model, "senti_vocab.txt"),
                          os.path.join(args.bert_model, "RoBERTa_Sentiment_kor"))

    dataset = NSMCDataSet("test", tokenizer, max_seq_length=args.max_seq_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            collate_fn=dataset.collate_fn)

    device = torch.device(type="cuda", index=args.gpu_index)

    model = RobertaForSequenceClassification()
    model_path = os.path.join('checkpoints/yaho/', '*.ckpt')
    model_path_list = glob.glob(model_path)
    for path in model_path_list:
        model.load_state_dict(state_dict=torch.load(path, map_location=torch.device('cpu')), strict=False)

        model.to(device)
        model.eval()

        match = 0
        progress = 0

        pbar = tqdm(dataloader, disable=args.no_display, desc="Eval")
        for batch in pbar:
            input_ids, attention_mask, labels = batch

            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).cuda(),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).cuda()
            }

            with torch.no_grad():
                logits = model(**inputs)

            labels = torch.tensor(labels, dtype=torch.float).cuda()

            match_seq = (logits.view(-1) >= 0.0) == (labels.view(-1) == 1)
            match += match_seq.sum().item()
            progress += labels.size(0)

            pbar.update()
            pbar.set_postfix(
                {"state_dict": path, "accuracy": f"{100.0 * match / progress:.2f}"}
            )
        pbar.close()
        log_file = open('./output/10^5step_log.txt', 'a')
        log_file.write("state_dict : " + path + "accuracy :" + str(100 * match / progress) + '\n')
        log_file.close()
        print({"state_dict": path, "accuracy": f"{100 * match / progress:.2f}"})
