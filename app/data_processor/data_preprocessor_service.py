import re
import pandas as pd
from datasets import Dataset


class DataPreprocessorService:

    @staticmethod
    def clean_text(text):
        SPECIAL_TOKENS = ['<s>', '</s>', '<pad>', '<unk>', '<sep>', '<cls>', '<mask>']

        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[“”‘’]', '"', text)  # Normalize quotes

        # Remove or escape special tokens
        for token in SPECIAL_TOKENS:
            text = text.replace(token, "")

        return text

    @staticmethod
    def preprocess_dataset(df: pd.DataFrame, direction="en>>hi"):
        df.dropna(inplace=True)

        # Clean both languages
        df['en'] = df['en'].apply(DataPreprocessorService.clean_text)
        df['hi'] = df['hi'].apply(DataPreprocessorService.clean_text)

        # Remove empty or too long sentences
        df = df[(df['en'].str.len() > 1) & (df['hi'].str.len() > 1)]
        df = df[(df['en'].str.len() < 200) & (df['hi'].str.len() < 200)]

        # Lowercase English (optional, model-dependent)
        df['en'] = df['en'].str.lower()

        # Format inputs with language tag
        if direction == "en>>hi":
            df_formatted = pd.DataFrame({
                "input_text": df["en"].apply(lambda x: f"eng_Latn hin_Deva {x}"),
                "target_text": df["hi"].apply(lambda x: f"eng_Latn hin_Deva {x}")
            })
        elif direction == "hi>>en":
            df_formatted = pd.DataFrame({
                "input_text": df["hi"].apply(lambda x: f"hin_Deva eng_Latn {x}"),
                "target_text": df["en"].apply(lambda x: f"hin_Deva eng_Latn {x}")
            })
        elif direction == "both":
            en_hi = pd.DataFrame({
                "input_text": df["en"].apply(lambda x: f"eng_Latn hin_Deva {x}"),
                "target_text": df["hi"].apply(lambda x: f"eng_Latn hin_Deva {x}")
            })
            hi_en = pd.DataFrame({
                "input_text": df["hi"].apply(lambda x: f"hin_Deva eng_Latn {x}"),
                "target_text": df["en"].apply(lambda x: f"hin_Deva eng_Latn {x}")
            })
            df_formatted = pd.concat([en_hi, hi_en], ignore_index=True)
        else:
            raise ValueError("Unsupported direction")

        assert df_formatted['input_text'].str.startswith('eng_Latn hin_Deva').any() or \
               df_formatted['input_text'].str.startswith('hin_Deva eng_Latn').any(), \
            "Missing or incorrect language tags in input_text!"

        assert df_formatted['target_text'].str.startswith('eng_Latn hin_Deva').any() or \
               df_formatted['target_text'].str.startswith('hin_Deva eng_Latn').any(), \
            "Missing or incorrect language tags in target_text!"

        return Dataset.from_pandas(df_formatted)