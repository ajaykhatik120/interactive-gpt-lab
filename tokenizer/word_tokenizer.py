import re
from collections import Counter


class WordTokenizer:

    def __init__(self):
        self.special_tokens = ["<PAD>", "<UNK>", "<EOS>"]

        self.stoi = {}
        self.itos = {}

        self.vocab_size = 0

    def tokenize(self, text):
        """
        Split text into words and punctuation.
        """

        text = text.lower()

        tokens = re.findall(
            r"\w+(?:['’]\w+)*|[^\w\s]",
            text
        )

        return tokens

    def build_vocabulary(self, text):
        """
        Build vocabulary from training text.
        """

        tokens = self.tokenize(text)

        unique_words = sorted(set(tokens))

        vocabulary = self.special_tokens + unique_words

        self.stoi = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        self.itos = {
            index: token
            for token, index in self.stoi.items()
        }

        self.vocab_size = len(self.stoi)

    def encode(self, text):
        """
        Convert text into integer token IDs.
        """

        tokens = self.tokenize(text)

        token_ids = [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]

        return token_ids

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """

        tokens = [
            self.itos[token_id]
            for token_id in token_ids
        ]

        return " ".join(
            token for token in tokens
            if token not in self.special_tokens
        )
    def encode_corpus(self, text):
         
         """
         Encode sentences and append EOS after each sentence.
         """

         sentences = re.split(r'(?<=[.!?])\s+', text.strip())

         token_ids = []

         for sentence in sentences:

             if not sentence.strip():
                 continue

             token_ids.extend(self.encode(sentence))

             token_ids.append(self.stoi["<EOS>"])


         return token_ids