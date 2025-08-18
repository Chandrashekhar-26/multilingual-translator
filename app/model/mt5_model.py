# import os
# import pandas as pd
# import torch
# from tqdm import tqdm
# from datasets import load_dataset, DatasetDict
# from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
# from app.data_processor import DataProcessorService
#
# tqdm.pandas()
#
#
# class MT5TranslationModel:
#     model = None
#     tokenizer = None
#     model_save_dir = "./saved_models/mt5_translation"
#
#     def __init__(self):
#         os.makedirs(self.model_save_dir, exist_ok=True)
#
#         model_name = "google/mt5-small"  # google/mt5-base
#         print(f"Loading model: {model_name}")
#         self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
#         self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
#
#         # Ensure tokenizer has pad_token
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         # Use GPU if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#
#         # Load dataset
#         dataset = load_dataset("cfilt/iitb-english-hindi")
#         train_dataset = self.__flatten_dataset(dataset['train'], 0.02)
#         val_dataset = self.__flatten_dataset(dataset['validation'])
#         test_dataset = self.__flatten_dataset(dataset['test'])
#
#         # self.train(train_dataset, val_dataset, test_dataset)
#
#     def __flatten_dataset(self, dataset, frac=None):
#         df = dataset.to_pandas()
#         if frac:
#             df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
#         df[['en', 'hi']] = df['translation'].progress_apply(pd.Series)
#         df.drop(columns='translation', inplace=True)
#         return df
#
#     def train(self, train_df, val_df, test_df):
#         # Preprocess (prefix-based format)
#         p_train = DataProcessorService.preprocess_data(train_df, direction="both", format="mt5")
#         p_val = DataProcessorService.preprocess_data(val_df, direction="both", format="mt5")
#
#         dataset = DatasetDict({
#             "train": p_train,
#             "validation": p_val
#         })
#
#         tokenized = dataset.map(lambda x: DataProcessorService.tokenize_data(x, self.tokenizer), batched=True)
#
#         training_args = TrainingArguments(
#             output_dir=self.model_save_dir,
#             per_device_train_batch_size=4,
#             gradient_accumulation_steps=4,
#             per_device_eval_batch_size=4,
#             learning_rate=3e-5,
#             num_train_epochs=3,
#             logging_dir="./logs",
#             evaluation_strategy="epoch",
#             save_strategy="epoch",
#             fp16=torch.cuda.is_available(),
#         )
#
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=tokenized["train"],
#             eval_dataset=tokenized["validation"]
#         )
#
#         trainer.train()
#         trainer.save_model(self.model_save_dir)
#         self.tokenizer.save_pretrained(self.model_save_dir)
#
#         print("Training complete & model saved.")
#
#     def translate(self, input_text: str, src_lang: str, tgt_lang: str) -> str:
#         device = self.model.device
#
#         prefix = f"translate {src_lang} to {tgt_lang}: "
#         input_ids = self.tokenizer(prefix + input_text, padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids.to(device)
#
#         with torch.no_grad():
#             outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
#
#         translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         return translation
#
# mt5_model = MT5TranslationModel()
