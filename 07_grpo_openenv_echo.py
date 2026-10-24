import os
import torch
from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = os.environ.get("MODEL","Qwen/Qwen3-0.6B")
ENV_URL = os.environ.get("OPENENV_URL","https://openenv-echo.env.hf.space")

class EchoToolEnv:
    def __init__(self):
        # New client/session per rollout; the server stays remote
        self.env = EchoEnv(base_url=ENV_URL)
        self.reward = 0.0

    def reset(self, **kwargs) -> str|None:
        self.reward = 0.0
        return None

    def echo(self,message:str)->str:
        """Send a message to the remote Echo environment
 
        Args: 
            message: Text that the remote environment should echo.

        Returns: 
            The echoed message supplied by the server.
        """
        result = self.env.step(EchoAction(message=message))
        self.reward = float(result.observation.reward or 0.0)
        return result.observation.echoed_message

def environment_reward(environments, **kwargs)->list[float]:
    # Called AFTER the model has sent messages to OpenEnv.
    return [env.reward for env in environments]

def main() -> None:
    prompts = [[{"role":"user", "content": "Use the echo tool with a useful short message, and read its output."}]] * 64
    dataset = Dataset.from_dict({"prompt":prompts})
    args = GRPOConfig(
        output_dir = "outputs/ch07_openenv",max_steps=15,
        per_device_train_batch_size = 1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=128,
        max_tool_calling_iterations = 3, learning_rate =5e-6,
        chat_template_kwargs={"enable_thinking":False},
        model_init_kwargs={"dtype":"bfloat16"},
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpoint=True, logging_steps=5,
        save_strategy="no", report_to="none", log_completions=True,
    )

    trainer = GRPOTrainer(
        model=MODEL, args=args, train_dataset = dataset,
        reward_funcs = environment_reward, environment_factory = EchoToolEnv,
        peft_config=LoraConfig(r=8,lora_alpha=16,target_modules = "all-linear"),
    )

    trainer.train()
    trainer.save_model("outputs/ch07_openenv/final_adapter")

if __name__ == "__main__":
    main()

