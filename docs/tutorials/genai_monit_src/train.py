import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOConfig, ORPOTrainer

from mlrun.execution import MLClientCtx


def train(
    context: MLClientCtx,
    dataset: str,
    base_model: str,
    new_model: str,
    device: str,
):
    import os

    transformers.logging.set_verbosity_warning()
    os.environ["HF_TOKEN"] = context.get_secret(key="HF_TOKEN")
    dataset = load_dataset(dataset, split="train").shuffle(seed=42)
    training_dataset = dataset.train_test_split(test_size=0.01)

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device,
        attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(model)

    orpo_args = ORPOConfig(
        # learning_rate=1e-3,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        max_length=1024,
        max_prompt_length=512,
        beta=0.2,
        # auto_find_batch_size=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=0.2,
        logging_steps=4,
        warmup_steps=2,
        output_dir="mlrun/banking-adapter2",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=new_model,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=training_dataset["train"],
        eval_dataset=training_dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # Apply training with evaluation:
    context.logger.info(f"training '{new_model}' based on '{base_model}'")
    trainer.train()
