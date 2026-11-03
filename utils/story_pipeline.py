import re
from tokenizer.word_tokenizer import WordTokenizer


def process_story(story_text):
    try:
        if not story_text or not story_text.strip():
            return {"success": False, "error": "Story cannot be empty."}

        # 1. Build tokenizer and vocabulary
        tokenizer = WordTokenizer()
        tokenizer.build_vocabulary(story_text)

        # 2. Convert story into token IDs
        token_ids = tokenizer.encode_corpus(story_text)

        # 3. Extract sentences
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", story_text.strip())
            if sentence.strip()
        ]

        # 4. Return processed results
        return {
            "success": True,
            "tokenizer": tokenizer,
            "token_ids": token_ids,
            "vocab_size": tokenizer.vocab_size,
            "tokens": tokenizer.tokenize(story_text),
            "sentences": sentences,
            "sentence_count": len(sentences),
            "token_count": len(token_ids),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
