import os
import random
import re
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")

def multiply(a: int, b: int) -> int:
    """Return the exact product of two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The product a times b.
    """
    return a * b

def answer_reward(completions, answer, **kwargs) -> list[float]:
    """Score numeric correctness, not compliance with an ANSWER=... format.

    Accepts final answers such as 128, ANSWER=128, Answer: <128>,
    \boxed{128}, and '16 x 8 = 128'. Avoids rewarding a number merely
    because it appears somewhere in a longer response when an explicit
    final-answer label is present.
    """
    number_pattern = r"(?<![\w.])-?\d+(?![\w.])"
    scores = []

    for completion, expected in zip(completions, answer):
        text = as_text(completion)
        predicted = None

        # Prefer the final explicitly labelled answer, in any format.
        for line in reversed(text.splitlines()):
            labels = list(re.finditer(r"\banswer\b", line, flags=re.IGNORECASE))
            if not labels:
                continue
            tail = line[labels[-1].end():]
            nums = re.findall(number_pattern, tail)
            if nums:
                # Placeholder + calculation: "Answer: <integer> = 16 x 8 = 128"
                # Also covers "Answer = 16 x 8 = 128".
                if re.search(r"<\s*integer\s*>", tail, re.IGNORECASE) or tail.count("=") >= 2:
                    predicted = int(nums[-1])
                else:
                    predicted = int(nums[0])
                break

        # No explicit answer label: final integer anywhere in the response.
        if predicted is None:
            nums = re.findall(number_pattern, text)
            predicted = int(nums[-1]) if nums else None

        scores.append(float(predicted == int(expected)))

    return scores

def main() -> None:
    rng = random.Random(2026)
    records = []
    for _ in range(96):
        a,b = rng.randint(8,99), rng.randint(3,25)
        records.append({
            "prompt": [{"role":"user", "content": f"Use the multiply tool to calculate {a} * {b}. Read the tool result and finish with ANSWER=<integer>."}],
            "answer": a*b,
        })
    dataset = Dataset.from_list(records)
    print("Check multiply tool:",multiply(7,8))
    args = GRPOConfig(
        output_dir = "outputs/ch05_tools", max_steps = 25,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=128,
        max_tool_calling_iterations=3,
        learning_rate = 5e-6,
        chat_template_kwargs = {"enable_thinking": False},
        model_init_kwargs = {"dtype": "bfloat16"},
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing = True, logging_steps=5, log_completions=True,
        save_strategy="no", report_to="none",
    )

    trainer= GRPOTrainer(
        model=MODEL, args=args, train_dataset=dataset,
        tools=[multiply],reward_funcs = answer_reward,
        peft_config= LoraConfig(r=8,lora_alpha=16,target_modules="all-linear"),
    )

    trainer.train()
    trainer.save_model("outputs/ch05_tools/final_adapter")

if __name__ == "__main__":
    main()