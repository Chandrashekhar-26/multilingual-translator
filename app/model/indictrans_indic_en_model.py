import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
from app.data_processor import DataProcessorService, evaluator
from datasets import load_dataset, DatasetDict
from IndicTransToolkit import IndicProcessor
from tqdm import tqdm
import os

tqdm.pandas()


class IndicTransIndicEnModel:
    model = None
    tokenizer = None
    model_save_dir = "./saved_models/indictrans2-indic-en-finetuned-v1"
    evaluation_result = {}

    ip = IndicProcessor(inference=True)

    def __init__(self):
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Load tokenizer and model from local disk
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_save_dir, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_save_dir, trust_remote_code=True)
        except:
            pass

        if self.model is None or self.tokenizer is None:
            # check gpu
            gpu_available = torch.device(True if torch.cuda.is_available() else False)

            # Load 4-bit quantized model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # Load model and tokenizer
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
            print(f"Downloading model from Hugging Face: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if gpu_available:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True, torch_dtype="auto")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # train on hindi-english dataset
        # dataset = load_dataset("cfilt/iitb-english-hindi") # Load dataset
        # train_dataset = self.__flatten_dataset(dataset['train'],  'hi', 'en', 0.02)
        # validation_dataset = self.__flatten_dataset(dataset['validation'], 'hi', 'en')
        # test_dataset = self.__flatten_dataset(dataset['test'], 'hi', 'en')

        # train
        # self.train(train_dataset, validation_dataset, test_dataset, 'hi', 'en', 'hin_Deva eng_Latn')


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
                df = df['translation'].progress_apply(pd.Series)

                # df[[src_lang, trgt_lang]] = df['translation'].progress_apply(pd.Series)
                # df.drop(columns='translation', inplace=True)

        return df

    def train(self, train_dataset, validation_dataset, test_dataset, src_lang, trgt_lang, prefix):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        # preprocess dataset
        p_train_dataset, p_validation_dataset, p_test_dataset = None,None, None

        if train_dataset is not None and not train_dataset.empty:
            p_train_dataset = DataProcessorService.preprocess_data(train_dataset, src_lang, trgt_lang, prefix)

        if validation_dataset is not None and not validation_dataset.empty:
            p_validation_dataset = DataProcessorService.preprocess_data(validation_dataset, src_lang, trgt_lang, prefix)

        if test_dataset is not None and not test_dataset.empty:
            p_test_dataset = DataProcessorService.preprocess_data(test_dataset, src_lang, trgt_lang, prefix)

        # prepare dataset (Bidirectional Handle)
        dataset_prepped = DatasetDict({
            "train": p_train_dataset,
            "validation": p_validation_dataset
        })

        tokenized = dataset_prepped.map(lambda batch: DataProcessorService.tokenize_data(batch, self.tokenizer), batched=True)

        # Freeze Layers for training stability, Leverage Pretrained Knowledge, Prevent Overfitting and reduce Compute and Memory usage
        # self.freeze_layers(self.model)

        # prepare training args
        training_args = TrainingArguments(
            output_dir=f'{self.model_save_dir}',
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
        print('Model Trained Successfully')

        # Save model and tokenizer
        trainer.save_model(self.model_save_dir)
        # self.model.save_pretrained(self.model_save_dir)
        # self.tokenizer.save_pretrained(self.model_save_dir)
        print('Saved trained model')

        # evaluate
        self.evaluation_result = self.evaluate(p_test_dataset, src_lang, trgt_lang)

    def evaluate(self, test_dataset, src_lang, tgt_lang):
        if test_dataset is None or test_dataset.empty:
            print("Test dataset is empty or not provided.")
            return

        print("Evaluating model...")

        input_texts = test_dataset[src_lang]
        references = test_dataset[tgt_lang]

        predictions = []
        for text in tqdm(input_texts, desc="Generating translations"):
            translation = self.translate(text, src_lang=src_lang, tgt_lang=tgt_lang)
            predictions.append(translation[0])

        # Evaluate
        metrics = evaluator.evaluate_translations(predictions, references, verbose=True)

        return metrics

    # def freeze_layers(self, model):
    #     # Freeze all parameters first
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     # Unfreeze last 2 encoder layers
    #     for layer in model.model.encoder.layers[-2:]:
    #         for param in layer.parameters():
    #             param.requires_grad = True
    #
    #     # Unfreeze last 2 decoder layers
    #     for layer in model.model.decoder.layers[-2:]:
    #         for param in layer.parameters():
    #             param.requires_grad = True
    #
    #     # Unfreeze LM head
    #     if hasattr(model.model, "lm_head"):
    #         for param in model.model.lm_head.parameters():
    #             param.requires_grad = True
    #
    #     # Optional: unfreeze embeddings if needed
    #     if hasattr(model.model, "shared"):
    #         for param in model.model.shared.parameters():
    #             param.requires_grad = True

    def translate(self, input_text: str, src_lang: str, tgt_lang: str, max_length: int = 256) -> str:
        device = self.model.device

        # Format the input in the expected format for IndicTrans2
        input_text = input_text.strip()

        # Check if input is Latin and transliterate if needed
        input_text = DataProcessorService.transliterate_text(input_text, src_lang, tgt_lang)

        batch = self.ip.preprocess_batch(
            [input_text],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        # Tokenize input
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=max_length
        ).to(device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(**inputs,use_cache=False, min_length=0, max_length=max_length, num_beams=5, num_return_sequences=1)

        # Decode output
        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Postprocess the translations, including entity replacement
        translations = self.ip.postprocess_batch(outputs, lang=tgt_lang)

        return translations


indic_trans_indic_en_model = IndicTransIndicEnModel()