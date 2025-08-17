import pandas as pd
from .data_preprocessor_service import DataPreprocessorService


class DataProcessorService:

    @staticmethod
    def preprocess_data(df: pd.DataFrame, direction: str):
        return DataPreprocessorService.preprocess_dataset(df, direction)

    # @staticmethod
    # def make_bidirectional(hf_dataset):
    #     df = hf_dataset.to_pandas()
    #     df.dropna(inplace=True)
    #
    #     en_hi = pd.DataFrame({
    #         "input_text": "en>>hi " + df["en"],
    #         "target_text": df["hi"]
    #     })
    #
    #     hi_en = pd.DataFrame({
    #         "input_text": "hi>>en " + df["hi"],
    #         "target_text": df["en"]
    #     })
    #
    #     combined_df = pd.concat([en_hi, hi_en], ignore_index=True)
    #     return Dataset.from_pandas(combined_df)

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

    # @staticmethod
    # def prepare_dataset(dataset):
    #
    #     train_bidirectional = DataProcessorService.make_bidirectional(dataset["train"])
    #     val_bidirectional = DataProcessorService.make_bidirectional(dataset["validation"]) if "validation" in dataset else None
    #     test_bidirectional = DataProcessorService.make_bidirectional(dataset["test"]) if "test" in dataset else None
    #
    #     prepared_dataset = DatasetDict({
    #         "train": train_bidirectional,
    #         **({"validation": val_bidirectional} if val_bidirectional else {}),
    #         **({"test": test_bidirectional} if test_bidirectional else {})
    #     })
    #
    #     return prepared_dataset
