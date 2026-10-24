#Chapter 4: GRPO on problems scored by a deterministic answer checker

import os
import random
import re
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")

def as_text(completion)->str:
    if isinstance(completion,str):
        return completion
    if isinstance(completion,list):
        return "\n".join(str(m.get("content","")) for m in completion if isinstance(m,dict))

def answer_reward(completions, answer, **kwargs):
    scores = []

    for completion, expected in zip(completions, answer):
        text = as_text(completion)

        # Preferred format
        strict = re.findall(
            r"ANSWER\s*=\s*(-?\d+)\b",
            text
        )

        # Alternative LaTeX format
        boxed = re.findall(
            r"\\boxed\{(-?\d+)\}",
            text
        )

        if strict:
            score = 1.0 if int(strict[-1]) == int(expected) else 0.0
        elif boxed:
            score = 0.8 if int(boxed[-1]) == int(expected) else 0.0
        else:
            score = 0.0

        scores.append(score)

    return scores

def main() -> None:
    rng = random.Random(2026)
    records = [] 
    for _ in range(128):
        a,b = rng.randint(1,20), rng.randint(1,12)
        records.append({
            "prompt": [{"role":"user", "content": f"Multiply {a} by {b}. Respond with ANSWER=<integer>."}],
            "answer": a*b,
        })
    dataset = Dataset.from_list(records)
    assert answer_reward(["ANSWER=12","ANSWER=4"],[12,12]) == [1.0,0.0]
    args = GRPOConfig(
        output_dir = "outputs/ch04_grpo", max_steps = 200,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=96,
        learning_rate = 5e-6, beta=0.0,
        chat_template_kwargs = {"enable_thinking": False},
        model_init_kwargs = {"dtype": "bfloat16"},
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing = True, logging_steps=5,
        save_strategy="no", report_to="none",
        log_completions = True,
    )

    trainer= GRPOTrainer(
        model=MODEL, args=args, train_dataset=dataset,
        reward_funcs = answer_reward,
        peft_config= LoraConfig(r=8,lora_alpha=16,target_modules="all-linear"),
    )

    trainer.train()
    trainer.save_model("outputs/ch04_grpo/final_adapter")

if __name__ == "__main__":
    main()