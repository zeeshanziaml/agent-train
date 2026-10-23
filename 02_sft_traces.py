import os
import random
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")

def make_trace(a: int, b: int)->dict:
    # The calls and tool observations are previously recorded; no sandbox is run
    return {"messages": [
        {"role":"user", "content": f"Compute {a} + {b}. Use the add tool, then answer."},
        {"role":"assistant", "tool_calls": [{"type":"function", "function":{"name":"add", "arguments":{"a":a, "b":b}}}]},
        {"role":"tool", "name": "add", "content": str(a+b)},
        {"role":"assistant", "content": f"ANSWER={a+b}"},
    ]}

def main() -> None:
    rng = random.Random(2026)
    records = [make_trace(rng.randrange(-30,31), rng.randrange(-30,31)) for _ in range(96)]
    dataset = Dataset.from_list(records)
    print("First recorded trace:", records[0])
    assert len(dataset) == 96
    args = SFTConfig(
        output_dir = "outputs/ch02_sft",max_steps=30,
        per_device_train_batch_size=1,gradient_accumulation_steps=4,
        learning_rate=1e-4, max_length=384,
        assistant_only_loss = True, #ignore user and tool-return tokens in loss
        model_init_kwargs = {"dtype":"bfloat16"} if torch.cuda.is_available() else {},
        bf16= torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing = True, logging_steps = 5,
        save_strategy="no", report_to="none",
    )

    trainer = SFTTrainer(
        model=MODEL, args=args, train_dataset = dataset,
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules = "all-linear"),
    )

    trainer.train()
    trainer.save_model("outputs/ch02_sft/final_adapter")
    print("Saved SFT LoRA adapter.")

if __name__ == "__main__":
    main()