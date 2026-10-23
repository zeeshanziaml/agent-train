#Chapter3: student samples; teacher grades token distributions
import os
import random
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DistillationConfig, DistillationTrainer

STUDENT = os.environ.get("MODEL","Qwen/Qwen3-0.6B")
TEACHER = os.environ.get("MODEL","Qwen/Qwen3-1.7B")

def main() -> None:
    rng = random.Random(2026)
    # Prompts only! DistillationTrainer asks STUDENT to complete them.
    prompts = [
        {"prompt": [{"role": "user", "content": f"Calculate {a}+{b}. Reply with ANSWER=<integer>."}]}

        for a,b in [(rng.randrange(-90,91), rng.randrange(-90,91)) for _ in range(96)]
    ]
    dataset = Dataset.from_list(prompts)
    args = DistillationConfig(
        output_dir = "outputs/ch03_distill",max_steps=15,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate = 1e-5, max_completion_length=48,
        chat_template_kwargs={"enable_thinking":False},
        model_init_kwargs = {"dtype":"bfloat16"},
        teacher_model_init_kwargs = {"dtype":"bfloat16"},
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, logging_steps=5,
        save_strategy="no", report_to="none",
    )
    trainer = DistillationTrainer(
        model=STUDENT, teacher_model = TEACHER, train_dataset = dataset,
        args = args,
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"),
    )

    trainer.train()
    trainer.save_model("outputs/ch03_distill/final_adapter")
    print("Finished student-on-policy distillation.")


if __name__ == "__main__":
    main()