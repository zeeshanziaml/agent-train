"""Chapter 7: GRPO with a LOCAL OpenEnv Echo server.

Run the Echo server in another terminal, then:
    export OPENENV_URL=http://127.0.0.1:8000
    python 07_grpo_openenv_echo_local.py --check-env
    python 07_grpo_openenv_echo_local.py

The Echo server supplies tool observations, but its built-in reward is 0.
This chapter therefore uses a custom reward: a successful exact echo and
an assistant final answer containing the echoed target.
"""

import argparse
import os
import random

import torch
from datasets import Dataset
from echo_env import CallToolAction, EchoEnv
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
ENV_URL = os.environ.get("OPENENV_URL", "http://127.0.0.1:8000")


def extract_echo_text(value) -> str:
    """Extract echoed text from common OpenEnv/MCP result formats."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("data", "result", "text"):
            if key in value:
                return extract_echo_text(value[key])
        if "content" in value:
            return extract_echo_text(value["content"])
    if isinstance(value, list):
        pieces = [extract_echo_text(item) for item in value]
        return "\n".join(piece for piece in pieces if piece)
    if hasattr(value, "data"):
        return extract_echo_text(value.data)
    return "" if value is None else str(value)


def last_assistant_text(completion) -> str:
    """Read the final assistant response, excluding tool observations."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        # Require the rollout to END with a real assistant response.
        # A tool call without a follow-up assistant answer is not enough.
        last = completion[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            content = last.get("content")
            return content if isinstance(content, str) else ""
    return ""


class EchoToolEnv:
    """One independent OpenEnv client/session for an agent rollout."""

    def __init__(self):
        self._context = None
        self._client = None
        self.target = None
        self.echo_succeeded = False

    def reset(self, target=None, **kwargs) -> None:
        """Called by GRPO at the beginning of every rollout."""
        if self._client is None:
            ctx = EchoEnv(base_url=ENV_URL).sync()
            client = ctx.__enter__()  # Connect and create the remote session
            self._context, self._client = ctx, client

        self._client.reset()
        self.target = target  # The dataset's per-prompt target column
        self.echo_succeeded = False
        return None

    def echo(self, message: str) -> str:
        """Call the remote environment's echo_message tool.

        Args:
            message: Text to send to the remote Echo server.

        Returns:
            The text returned by the server.
        """
        if self._client is None:
            raise RuntimeError("Call reset() before echo().")

        step = self._client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": message},
            )
        )
        observation = step.observation
        if getattr(observation, "error", None):
            raise RuntimeError(f"OpenEnv echo failed: {observation.error}")

        echoed = extract_echo_text(observation.result)
        if message == self.target and echoed == self.target:
            self.echo_succeeded = True
        if os.environ.get("DEBUG_ECHO") == "1":
            print(f"[ECHO] sent={message!r} received={echoed!r} target={self.target!r}", flush=True)
        return echoed

    def _close(self) -> None:
        if self._context is not None:
            ctx, self._context = self._context, None
            self._client = None
            ctx.__exit__(None, None, None)

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass


def environment_reward(environments, completions, **kwargs) -> list[float]:
    """Reward only a real successful echo plus a final assistant response."""
    rewards = []
    for env, completion in zip(environments, completions):
        final = last_assistant_text(completion)
        correct = bool(env.echo_succeeded and env.target and env.target in final)
        rewards.append(float(correct))
    return rewards


def check_environment() -> None:
    """Test the local server before loading Qwen or starting training."""
    sample = "Hello from Chapter 7"
    print(f"Checking OpenEnv Echo server at {ENV_URL} ...", flush=True)
    try:
        with EchoEnv(base_url=ENV_URL).sync() as client:
            client.reset()
            step = client.step(
                CallToolAction(
                    tool_name="echo_message", arguments={"message": sample}
                )
            )
            echoed = extract_echo_text(step.observation.result)
            print(f"Sent:     {sample!r}")
            print(f"Received: {echoed!r}")
            if echoed != sample:
                raise RuntimeError(f"Echo test returned an unexpected result: {step.observation.result!r}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot use the Echo server at {ENV_URL}. Start it in another "
            "RunPod terminal and run this --check-env test again."
        ) from exc
    print("OpenEnv Echo connection: OK")


def main() -> None:
    # Fail early with an actionable error instead of downloading the model first.
    check_environment()

    rng = random.Random(2026)
    rows = []
    for i in range(64):
        target = f"OpenEnv practice {i}: code {rng.randint(100, 999)}"
        rows.append({
            "prompt": [{
                "role": "user",
                "content": (
                    "Use the echo tool to send exactly the following text. "
                    "Read the tool reply, then include the echoed text in your final answer.\n"
                    f"Text: {target}"
                ),
            }],
            "target": target,
        })
    dataset = Dataset.from_list(rows)

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
    parser = argparse.ArgumentParser(description="Chapter 7 GRPO + local OpenEnv Echo")
    parser.add_argument("--check-env", action="store_true", help="Test Echo connection without training")
    cli = parser.parse_args()
    if cli.check_env:
        check_environment()
    else:
        main()
