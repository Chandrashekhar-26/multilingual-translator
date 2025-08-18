import pandas as pd
from .data_preprocessor_service import DataPreprocessorService


class DataProcessorService:

    @staticmethod
    def preprocess_data(df: pd.DataFrame, direction: str):
        return DataPreprocessorService.preprocess_dataset(df, direction)

    @staticmethod
    def tokenize_data(batch, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(batch["target_text"], truncation=True, padding="max_length", max_length=128).input_ids
        model_inputs["labels"] = [
            [(lbl if lbl != tokenizer.pad_token_id else -100) for lbl in label] for label in labels
        ]
        return model_inputs
