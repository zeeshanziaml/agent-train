from dataclasses import dataclass, field

@dataclass
class RepairSandbox:
    source:str = "def add(a,b):\n   return a-b\n"
    transcript: list[dict] = field(default_factory=list)

    def call_tool(self, name:str, argument:str="")-> str:
        if name == "read_file":
            observation = self.source
        elif name == "replace_file":
            self.source = argument
            observation = "File updated"
        elif name == "run_tests":
            namespace = {}
            try:
                exec(self.source,namespace)
                ok = all(namespace["add"](a,b) == a+b
                         for a,b in [(-2,5),(0,7),(13,19)])
                observation = "3/3 passed" if ok else "tests failed"
            except Exception as exc:
                observation = f"error: {type(exc).__name__}"
        else:
            raise ValueError(f"Unknown tool: {name}")
        self.transcript.append({"role":"tool","name":name,"content":observation})

        return observation

    def reward(self)->float:
        return float(self.call_tool("run_tests")=="3/3 passed")

def scripted_model_policy(history: list[dict])->tuple[str,str]:
    tools = [h["name"] for h in history if h["role"] == "tool"]
    if "read_file" not in tools:
        return "read_file", ""
    if "replace_file" not in tools:
        return "replace_file", "def add(a,b):\n     return a+b\n"
    return "stop",""

def run_episode() -> tuple[float,list[dict]]:
    env = RepairSandbox()
    history = [{"role":"user", "content":"Fix add() so all tests pass."}]
    for _ in range(5):
        tool,arg = scripted_model_policy(history)
        history.append({"role":"assistant","tool":tool,"argument":arg})
        if tool == "stop":
            break
        result = env.call_tool(tool,arg)
        history.append({"role":"tool","name":tool,"content":result})
    score = env.reward()
    return score,history

if __name__ == "__main__":
    score,trace = run_episode()
    for event in trace:
        print(event)
    print(f"FINAL REWARD: {score:.1f}")
    assert score == 1.0