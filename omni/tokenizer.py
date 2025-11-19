
import sentencepiece as spm
import os

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @classmethod
    def train_new(cls, text_path, out_model, vocab_size=32000, max_sentence_length=100000, input_sentence_size=100000000):
        """
        Train a new BPE tokenizer.
        
        Args:
            text_path: Path to training text file
            out_model: Output model path
            vocab_size: Vocabulary size
            max_sentence_length: Maximum sentence length in bytes (default: 100000)
            input_sentence_size: Maximum number of sentences to use (default: 10000000 for faster training)
                                 Set to 0 to use all sentences (slower but uses more data)
        """  
        train_params = {
            'input': text_path,
            'model_prefix': out_model.replace('.model',''),
            'vocab_size': vocab_size,
            'model_type': 'bpe',
            'character_coverage': 1.0,
            'max_sentence_length': max_sentence_length,
            'train_extremely_large_corpus': True
        }
        
        # Limit sentences for faster training (default: 10M sentences)
        if input_sentence_size > 0:
            train_params['input_sentence_size'] = input_sentence_size
            print(f"  Limiting to {input_sentence_size:,} sentences for faster training...")
        
        spm.SentencePieceTrainer.train(**train_params)
        return cls(out_model)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)
