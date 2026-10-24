"""Chapter 7: GRPO with the current OpenEnv Echo MCP client.

The Echo server itself returns zero reward. For teaching, this exercise instead
rewards the agent when it echoes the exact target message supplied in the prompt.

Requires the `echo_env` client from current OpenEnv (CallToolAction API) and
an accessible, running Echo server. This program has not been GPU-tested.
"""
import os
import random

import torch
from datasets import Dataset
from echo_env import CallToolAction, EchoEnv
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
ENV_URL = os.environ.get("OPENENV_URL", "https://openenv-echo-env.hf.space")


class EchoToolEnv:
    def __init__(self):
        self._context = None
        self._client = None
        self.target = None
        self.reward = 0.0

    def reset(self, target=None, **kwargs) -> None:
        """Reset a training episode and establish the remote Echo session."""
        if self._client is None:
            self._context = EchoEnv(base_url=ENV_URL).sync()
            self._client = self._context.__enter__()
        self._client.reset()
        self.target = target
        self.reward = 0.0
        return None

    def echo(self, message: str) -> str:
        """Send a message to the remote Echo environment and return its reply.

        Args:
            message: Text that the remote environment should echo.

        Returns:
            The message returned by the remote environment.
        """
        if self._client is None:
            raise RuntimeError("The environment must be reset before echoing.")

        step_result = self._client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": message},
            )
        )
        if getattr(step_result.observation, "error", None):
            raise RuntimeError(f"Echo tool failed: {step_result.observation.error}")

        response = step_result.observation.result
        echoed = response.data if hasattr(response, "data") else response
        echoed = str(echoed)

        # Training reward is defined locally: the Echo server's own reward is 0.
        self.reward = float(message == self.target and echoed == self.target)
        return echoed

    def __del__(self):
        # Private cleanup so TRL does not expose it as an agent-callable tool.
        ctx = getattr(self, "_context", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass


def environment_reward(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


def main() -> None:
    rng = random.Random(2026)
    examples = []
    for i in range(64):
        target = f"OpenEnv practice {i}: code {rng.randint(100, 999)}"
        examples.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Use the echo tool to send the exact text below. "
                            "Read the tool result before replying.\n"
                            f"Text: {target}"
                        ),
                    }
                ],
                "target": target,
            }
        )
    dataset = Dataset.from_list(examples)

    args = GRPOConfig(
        output_dir="outputs/ch07_openenv",
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        learning_rate=5e-6,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=5,
        log_completions=True,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=MODEL,
        args=args,
        train_dataset=dataset,
        reward_funcs=environment_reward,
        environment_factory=EchoToolEnv,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model("outputs/ch07_openenv/final_adapter")


if __name__ == "__main__":
    main()
