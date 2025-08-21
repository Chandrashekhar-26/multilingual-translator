import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch
from app.data_processor import DataProcessorService
from datasets import load_dataset, DatasetDict
from IndicTransToolkit import IndicProcessor
from tqdm import tqdm
import os

tqdm.pandas()


class IndicTransIndicIndicModel:
    model = None
    tokenizer = None
    model_save_dir = "./saved_models/indictrans2-indic-indic-finetuned-v1"

    ip = IndicProcessor(inference=True)

    def __init__(self):
        print('************* Init indic-indic')
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Load tokenizer and model from local disk if available
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_save_dir, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_save_dir, trust_remote_code=True)
        except:
            pass

        if self.model is None or self.tokenizer is None:
            # Load from Hugging Face
            model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
            print(f"Downloading model from Hugging Face: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Load a suitable Indic-Indic dataset for fine-tuning
        dataset = load_dataset("ai4bharat/samanantar", "mr")

        # train_dataset = self.__flatten_dataset(dataset['train'], 0.02)
        # validation_dataset = self.__flatten_dataset(dataset['validation'])
        # test_dataset = self.__flatten_dataset(dataset['test'])

        # self.train(train_dataset, validation_dataset, test_dataset)

    def __flatten_dataset(self, dataset, src_lang: str, trgt_lang: str, frac=None):
        df = None
        if dataset:
            df_full = dataset.to_pandas()
            if frac is not None:
                df = df_full.sample(frac=frac, random_state=42).reset_index(drop=True)
            else:
                df = df_full
            print(df.head(5))

            if 'translation' in df:
                df[[src_lang, trgt_lang]] = df['translation'].progress_apply(pd.Series)

            df.drop(columns='translation', inplace=True)

        return df

    def train(self, train_dataset, validation_dataset, test_dataset):
        p_train_dataset, p_validation_dataset, p_test_dataset = None, None, None

        if train_dataset is not None and not train_dataset.empty:
            p_train_dataset = DataProcessorService.preprocess_data(train_dataset, direction="indic-indic")

        if validation_dataset is not None and not validation_dataset.empty:
            p_validation_dataset = DataProcessorService.preprocess_data(validation_dataset, direction="indic-indic")

        if test_dataset is not None and not test_dataset.empty:
            p_test_dataset = DataProcessorService.preprocess_data(test_dataset, direction="indic-indic")

        dataset_prepped = DatasetDict({
            "train": p_train_dataset,
            "validation": p_validation_dataset
        })

        tokenized = dataset_prepped.map(lambda batch: DataProcessorService.tokenize_data(batch, self.tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir=self.model_save_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            num_train_epochs=3,
            do_eval=True,
            eval_steps=500,
            save_strategy="epoch",
            logging_dir="./logs",
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation", None),
        )

        trainer.train()
        print("Model Trained Successfully")

        trainer.save_model(self.model_save_dir)
        print("Saved trained model")

    def translate(self, input_text: str, src_lang: str, tgt_lang: str, max_length: int = 256) -> str:
        device = self.model.device

        input_text = input_text.strip()

        # Check if input is Latin and transliterate if needed
        input_text = DataProcessorService.transliterate_text(input_text, src_lang, tgt_lang)


        batch = self.ip.preprocess_batch(
            [input_text],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1
            )

        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        translations = self.ip.postprocess_batch(outputs, lang=tgt_lang)

        return translations


# Instantiate the model
indictrans_indic_indic_model = IndicTransIndicIndicModel()
