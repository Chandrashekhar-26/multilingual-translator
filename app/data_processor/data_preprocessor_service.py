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
    def preprocess_dataset(df: pd.DataFrame, src_lang: str, trgt_lang: str, prefix='eng_Latn hin_Deva'):
        df.dropna(inplace=True)

        # Clean both languages
        df[src_lang] = df[src_lang].apply(DataPreprocessorService.clean_text)
        df[trgt_lang] = df[trgt_lang].apply(DataPreprocessorService.clean_text)

        # Remove empty or too long sentences
        df = df[(df[trgt_lang].str.len() > 1) & (df[src_lang].str.len() > 1)]
        df = df[(df[trgt_lang].str.len() < 200) & (df[src_lang].str.len() < 200)]

        # Lowercase English (optional, model-dependent)
        if 'en' in df.columns and df['en'].notnull().any():
            df['en'] = df['en'].str.lower()

        # Format inputs with language tag
        df_formatted = pd.DataFrame({
            "input_text": df[src_lang].apply(lambda x: f"{prefix} {x}"),
            "target_text": df[trgt_lang].apply(lambda x: f"{prefix} {x}")
        })
        #
        # assert df_formatted['input_text'].str.startswith('eng_Latn hin_Deva').any() or \
        #        df_formatted['input_text'].str.startswith('hin_Deva eng_Latn').any(), \
        #     "Missing or incorrect language tags in input_text!"
        #
        # assert df_formatted['target_text'].str.startswith('eng_Latn hin_Deva').any() or \
        #        df_formatted['target_text'].str.startswith('hin_Deva eng_Latn').any(), \
        #     "Missing or incorrect language tags in target_text!"

        return Dataset.from_pandas(df_formatted)