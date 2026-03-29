import re
import json
# We tokenize SMILES strings using a regex-based tokenizer following https://arxiv.org/pdf/1711.04810 

SMILES_REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

PAD_TOKEN = "<PAD>"  # index 0
SOS_TOKEN = "<SOS>"  # index 1
EOS_TOKEN = "<EOS>"  # index 2
UNK_TOKEN = "<UNK>"  # index 3
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class SMILESTokenizer:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_size = 0
        self._pattern = re.compile(SMILES_REGEX)

    def build_vocab(self, smiles_iter):
        # Collect all unique tokens from an iterable of SMILES, then assign indices
        seen = set()
        for smi in smiles_iter:
            seen.update(self._pattern.findall(smi))
        vocab_tokens = SPECIAL_TOKENS + sorted(seen)
        self.token2idx = {t: i for i, t in enumerate(vocab_tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        self.vocab_size = len(vocab_tokens)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token2idx, f)

    def load(self, path):
        with open(path) as f:
            self.token2idx = json.load(f)
        self.idx2token = {int(i): t for t, i in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    def tokenize(self, smiles):
        # Split a SMILES string into a list of token strings
        return self._pattern.findall(smiles)

    def encode(self, smiles, add_sos=True, add_eos=True, max_len=None):
        # Convert a SMILES string to a list of integer indices
        tokens = self.tokenize(smiles)
        if max_len is not None:
            capacity = max_len - int(add_sos) - int(add_eos)
            tokens = tokens[:capacity]
        indices = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        if add_sos:
            indices = [SOS_IDX] + indices
        if add_eos:
            indices = indices + [EOS_IDX]
        return indices

    def decode(self, indices, strip_special=True):
        # Convert integer indices back to a SMILES string, stopping at EOS
        tokens = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            token = self.idx2token.get(idx, UNK_TOKEN)
            if strip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)