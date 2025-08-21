import pandas as pd
from .data_preprocessor_service import DataPreprocessorService
from indic_transliteration.sanscript import transliterate, DEVANAGARI, KANNADA, TAMIL, ITRANS, HK, IAST
import re

class DataProcessorService:

    def has_latin(text: str):
        return bool(re.search(r'[a-zA-Z]', text))

    def transliterate_text(text: str, src_lang: str, target_lang: str, scheme: str = "IAST") -> str:
        lang_map = {
            'hin_Deva': DEVANAGARI,
            'mar_Deva': DEVANAGARI,
            'kan_Knda': KANNADA,
            'tam_Taml': TAMIL,
        }

        # Maps scheme name to Sanscript schemes
        scheme_map = {
            'IAST': IAST,
            'ITRANS': ITRANS,
            'HK': HK,
        }

        if src_lang not in lang_map:
            return text

        target_script = lang_map[src_lang]

        # src lang has latin, transliterate to native script
        if DataProcessorService.has_latin(text):
            return transliterate(text, ITRANS, target_script)


        # # Input is already in native script, convert to Latin scheme (default IAST)
        # if scheme in scheme_map:
        #     return transliterate(text, target_script, scheme_map[scheme])

        return text

    @staticmethod
    def preprocess_data(df: pd.DataFrame, src_lang: str, trgt_lang: str, prefix: str = 'eng_Latn hin_Deva'):
        return DataPreprocessorService.preprocess_dataset(df, src_lang, trgt_lang, prefix)

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
