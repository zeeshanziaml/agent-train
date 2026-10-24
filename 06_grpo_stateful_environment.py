import os
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")

class NumberLineEnv:
    """One instance per rollout: state must NOT be shared across episodes."""
    def __init__(self):
        self.position = 0
        self.target = 3
        self.moves = 0
        self.invalid = 0

    def reset(self,target:int=3, **kwargs)->str | None:
        self.position = 0
        self.target = int(target)
        self.moves = 0
        self.invalid = 0
        return None # the supplied dataset already has the user prompt

    def move(self, step: int) -> str:
        """Move the agent exactly one unit left or right.

        Args:
            step: Must be +1 or -1. Other values are rejected.

        Returns:
            Your new position and how far you are from the goal.
        """
        self.moves += 1
        if step not in (-1,1) or self.moves > 5:
            self.invalid += 1
            return f"Invalid move. Position={self.position}; target={self.target}"
        self.position += step
        return f"Position={self.position}; remaining={self.target - self.position}"

    def get_reward(self)->float:
        # End-of-episode verifier. Tool responses do not reveal this scalar.
        return float(self.position == self.target and self.invalid==0)

def main() -> None:
    env = NumberLineEnv()
    env.reset(target=3)
    assert env.move(1).startswith("Position=1")
    assert env.move(1).startswith("Position=2")
    assert env.move(1).startswith("Position=3")
    assert env.get_reward() == 1.0

    prompt = [{"role":"user", "content":"Use the move tool to reach position +3, starting from 0. Each call must be step=+1 or step=-1. Check the observation after each move; stop at +3."}]
    # Identical task for every member of each group -> comparable rewards.
    dataset = Dataset.from_list([{"prompt":prompt,"target":3} for _ in range(64)])
    args = GRPOConfig(
        output_dir = "outputs/ch06_environment", max_steps =25,
        per_device_train_batch_size = 1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=200,
        max_tool_calling_iterations = 5, learning_rate = 5e-6,
        chat_template_kwargs = {"enable_thinking":False},
        model_init_kwargs = {"dtype":"bfloat16"},
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing = True, logging_steps=5,
        save_strategy="no", report_to="none",
    )

    trainer=GRPOTrainer(
        model=MODEL, args=args, train_dataset=dataset,
        environment_factory=NumberLineEnv, #CLASS, not a shared instance
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model("outputs/ch06_environment/final_adapter")

if __name__ == "__main__":
    main()