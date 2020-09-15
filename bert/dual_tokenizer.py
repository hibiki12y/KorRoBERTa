import os
import sentencepiece as spm
from collections import OrderedDict

def load_vocab(vocab_file) :
    vocab = OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class Tokenizer(object) :

    def __init__(self, vocab_file, bpe_file, do_lower_case=False, max_len=None):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))

        self.do_lower_case = do_lower_case
        self.UNK_token = "[UNK]"
        self.vocab = load_vocab(vocab_file)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len if max_len is not None else int(1e12)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("{}.model".format(bpe_file))

    def tokenize(self, text) :
        if self.do_lower_case :
            text = text.lower()

        tokens = []
        is_mchar = []
        for token in text.split(" "):
            bpe_tokens = self.sp.EncodeAsPieces(token)
            for bpe_token in bpe_tokens:
                if bpe_token == "<unk>" or bpe_token not in self.vocab:
                    bpe_token = self.UNK_token
                tokens.append(bpe_token)
                is_mchar.append(False)

        return tokens, is_mchar

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens :
            ids.append(self.vocab.get(token, self.vocab[self.UNK_token]))
        return ids

    def convert_ids_to_tokens(self, ids) :
        tokens = []
        for idx in ids :
            tokens.append(self.reverse_vocab[idx])
        return tokens

