import os
import random
import re
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")

def as_text(completion) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # Extract the last assistant message, not the tool's output
        assistant_messages = [
            m for m in completion
            if isinstance(m, dict)
            and m.get("role") == "assistant"
        ]

        if assistant_messages:
            return str(assistant_messages[-1].get("content") or "")

    return ""

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
    """
    Reward successful tool use.

    Reward = 1.0 if:
      1. The agent actually called a tool.
      2. The agent produced a final answer after the tool response.
      3. The final numerical answer is correct.

    Otherwise, reward = 0.0.
    """
    scores = []

    for completion, expected in zip(completions, answer):

        # A tool-use trajectory should contain structured messages.
        if not isinstance(completion, list):
            scores.append(0.0)
            continue

        used_tool = False
        final_text = None

        for message in completion:
            if not isinstance(message, dict):
                continue

            role = message.get("role")

            if role == "tool":
                # A real tool response was recorded.
                used_tool = True
                final_text = None

            elif role == "assistant" and used_tool:
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    final_text = content

        # No tool execution or no final assistant answer.
        if not used_tool or not final_text:
            scores.append(0.0)
            continue

        # Extract the final numeric answer, regardless of formatting.
        number_pattern = r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])"

        # Prefer a line explicitly mentioning "answer".
        answer_lines = [
            line for line in final_text.splitlines()
            if re.search(r"\banswer\b", line, re.IGNORECASE)
        ]

        search_text = answer_lines[-1] if answer_lines else final_text
        numbers = re.findall(number_pattern, search_text)

        predicted = int(numbers[-1]) if numbers else None

        # Reward only correct answers following actual tool use.
        reward = float(predicted == int(expected))
        scores.append(reward)

    return scores

def main() -> None:
    rng = random.Random(2026)
    records = []
    for _ in range(96):
        a,b = rng.randint(1000,9999), rng.randint(100,999)
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