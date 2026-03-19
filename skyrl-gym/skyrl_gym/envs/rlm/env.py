import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.tools.repl import PersistentREPL


DEFAULT_RLM_SYSTEM_PROMPT = """\
You are tasked with answering a query with associated context. You can access, \
transform, and analyze this context interactively in a REPL environment. You \
will be queried iteratively until you provide a final answer.

Your context is a string with {context_length} total characters.

The REPL environment is initialized with:
1. A `context` variable that contains important information about your query. \
You should check the content of the context variable to understand what you are \
working with. Make sure you look through it sufficiently as you answer your query.
{llm_query_section}\
3. The ability to use `print()` statements to view the output of your REPL code \
and continue your reasoning. You will only be able to see truncated outputs from \
the REPL environment{llm_query_hint}.

Use variables as buffers to build up your final answer. Make sure to explicitly \
look through the context in the REPL before answering your query.

Respond with Python code wrapped in ```python ... ``` blocks.

IMPORTANT: When you are done, set `Final` to your answer in the REPL:
```python
Final = "your answer here"
```
or assign a variable you have built up:
```python
Final = my_answer_variable
```

Think step by step carefully, plan, and execute this plan immediately in your \
response -- do not just say "I will do this". Output to the REPL environment \
as much as possible. Remember to explicitly answer the original query in your \
final answer.\
"""


@dataclass
class RLMEnvConfig:
    metadata_prefix_length: int = 500
    repl_timeout: float = 15.0
    max_sub_calls_per_episode: int = 20
    system_prompt: Optional[str] = None
    supplementary_system_prompt: Optional[str] = None


class RLMEnv(BaseTextEnv):
    """
    Recursive Language Model environment.

    The model sees three layers at init:

    1. **System prompt** — default RLM instructions explaining the REPL, the
       ``context`` variable, ``llm_query()``, and how to set ``Final``.
       Override via ``RLMEnvConfig.system_prompt``.

    2. **Prompt** (dataset ``prompt`` field) — the user question / task
       description, shown directly in the model's context window.

    3. **Context** (``extra_info["context_text"]``) — stored externally in the
       REPL as the ``context`` variable. The model must run code to read it.

    If ``context_text`` is not provided, the prompt content is used as both the
    direct message *and* the REPL context (backward-compatible behavior).

    The model generates Python code each turn to interact with ``context``,
    call ``llm_query()`` for sub-RLM delegation, and set ``Final`` when done.
    Only compact metadata about execution results is returned, keeping the
    context window manageable.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras.get("max_turns", 10)

        if isinstance(env_config, RLMEnvConfig):
            self.rlm_config = env_config
        elif isinstance(env_config, Mapping):
            self.rlm_config = RLMEnvConfig(**{k: v for k, v in env_config.items() if k in RLMEnvConfig.__dataclass_fields__})
        else:
            self.rlm_config = RLMEnvConfig()

        if extras.get("supplementary_system_prompt"):
            self.rlm_config.supplementary_system_prompt = extras["supplementary_system_prompt"]

        self.llm_query_fn: Optional[Callable] = extras.get("llm_query_fn")
        self.sub_call_count = 0
        self.repl: Optional[PersistentREPL] = None
        self.chat_history: ConversationType = []
        self._metrics: Dict[str, Any] = {}

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        extra_info = self.extras.get("extra_info", {}) if hasattr(self, "extras") else {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        user_prompt = self._extract_prompt_text(prompt)

        # Context text stored in the REPL as `context`.
        # If extra_info provides context_text, use that; otherwise fall back to
        # the prompt content (backward compat: prompt serves as both).
        context_text = extra_info.get("context_text") or user_prompt

        has_llm_query = self.llm_query_fn is not None

        namespace = {"Final": None}
        namespace["context"] = context_text

        if has_llm_query:
            call_limit = self.rlm_config.max_sub_calls_per_episode

            def guarded_llm_query(sub_prompt: str, max_tokens: int = 4096) -> str:
                if self.sub_call_count >= call_limit:
                    return f"[Error: sub-call limit of {call_limit} reached]"
                self.sub_call_count += 1
                return self.llm_query_fn(sub_prompt, max_tokens)

            namespace["llm_query"] = guarded_llm_query

        self.repl = PersistentREPL(namespace=namespace, timeout=self.rlm_config.repl_timeout)

        system_prompt = self._build_system_prompt(
            context_length=len(context_text),
            has_llm_query=has_llm_query,
        )

        init_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.chat_history = list(init_messages)
        return init_messages, {}

    def _build_system_prompt(self, context_length: int, has_llm_query: bool) -> str:
        template = self.rlm_config.system_prompt or DEFAULT_RLM_SYSTEM_PROMPT

        if has_llm_query:
            llm_query_section = (
                "2. A `llm_query(prompt, max_tokens=4096)` function that allows you "
                "to query a sub-LLM inside your REPL environment.\n"
            )
            llm_query_hint = ", so use `llm_query()` on variables you want to analyze semantically"
        else:
            llm_query_section = ""
            llm_query_hint = ""

        prompt = template.format(
            context_length=context_length,
            llm_query_section=llm_query_section,
            llm_query_hint=llm_query_hint,
        )
        if self.rlm_config.supplementary_system_prompt:
            prompt += "\n\n" + self.rlm_config.supplementary_system_prompt
        return prompt

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        done = self.turns >= self.max_turns
        code = self._extract_code(action)

        if code is None:
            obs_text = "[No code block found. Wrap your code in ```python ... ``` blocks.]"
            if not done:
                obs = [{"role": "user", "content": obs_text}]
            else:
                obs = []
            reward = self._get_reward(done)
            return BaseTextEnvStepOutput(observations=obs, reward=reward, done=done, metadata={})

        result = self.repl.execute(code)

        if self.repl.namespace.get("Final") is not None:
            done = True

        reward = self._get_reward(done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=self._build_metadata())

        obs_text = self._format_observation(result)
        obs = [{"role": "user", "content": obs_text}]
        self.chat_history.append(obs[0])
        return BaseTextEnvStepOutput(observations=obs, reward=reward, done=False, metadata=self._build_metadata())

    def _extract_prompt_text(self, prompt: ConversationType) -> str:
        """Extract the raw text content from the prompt messages."""
        parts = []
        for msg in prompt:
            if msg.get("content"):
                parts.append(msg["content"])
        return "\n".join(parts)

    def _extract_code(self, action: str) -> Optional[str]:
        """Extract Python code from markdown code blocks or <code> tags."""
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"<code>(.*?)</code>",
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _get_reward(self, done: bool) -> float:
        if not done:
            return 0.0

        final_val = self.repl.namespace.get("Final") if self.repl else None
        if final_val is None:
            return 0.0

        final_str = str(final_val).strip()
        gt_str = str(self.ground_truth).strip()

        if final_str == gt_str:
            return 1.0

        try:
            if abs(float(final_str) - float(gt_str)) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass

        if gt_str.lower() in final_str.lower():
            return 0.5

        return 0.0

    def _format_observation(self, result) -> str:
        """Format REPL result as a compact metadata observation."""
        prefix_len = self.rlm_config.metadata_prefix_length
        parts = ["[Execution Result]"]

        if result.error:
            err = result.error
            if len(err) > prefix_len:
                parts.append(f"Error (last {prefix_len} of {len(err)} chars): ...{err[-prefix_len:]}")
            else:
                parts.append(f"Error: {err}")
        else:
            stdout = result.stdout
            if stdout:
                if len(stdout) > prefix_len:
                    parts.append(f"stdout (first {prefix_len} of {len(stdout)} chars): \"{stdout[:prefix_len]}\"")
                else:
                    parts.append(f"stdout: \"{stdout}\"")

        # parts.append(f"Variables: {self.repl.summarize_namespace()}")
        return "\n".join(parts)

    def _build_metadata(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "sub_calls": self.sub_call_count,
        }

    def get_metrics(self) -> Dict[str, Any]:
        final_val = self.repl.namespace.get("Final") if self.repl else None
        return {
            "turns_used": self.turns,
            "sub_calls_made": self.sub_call_count,
            "final_value_set": final_val is not None,
            "reward": self._get_reward(True),
        }

    def close(self):
        self.repl = None


if __name__ == "__main__":
    import argparse
    from skyrl_gym.tools.llm_client import ExternalSubLLMClient

    parser = argparse.ArgumentParser(description="RLMEnv sandbox")
    parser.add_argument("--sub-model", default=None, help="Sub-LLM model name (e.g. gpt-4o-mini). Enables llm_query().")
    parser.add_argument("--sub-model-url", default=None, help="Base URL for sub-LLM (for local vLLM servers)")
    parser.add_argument("--sub-model-api-key", default=None, help="API key for sub-LLM (defaults to OPENAI_API_KEY)")
    args = parser.parse_args()

    llm_query_fn = None
    if args.sub_model:
        client = ExternalSubLLMClient(
            base_url=args.sub_model_url,
            model=args.sub_model,
            api_key=args.sub_model_api_key,
        )
        llm_query_fn = client.query
        print(f"[sandbox] llm_query() enabled with model={args.sub_model}")

    # prompt = direct instructions the model sees in full
    direct_prompt = "What is the capital of France? Extract it from the context."

    # context_text = large text stored externally in the REPL `context` variable
    context_text = (
        "The population of France is approximately 67 million people. "
        "The capital city is Paris. France is known for the Eiffel Tower, "
        "fine cuisine, and its contributions to art and philosophy."
    )

    env = RLMEnv(
        env_config=RLMEnvConfig(metadata_prefix_length=300, repl_timeout=30.0),
        extras={
            "reward_spec": {"ground_truth": "Paris"},
            "max_turns": 5,
            "extra_info": {"context_text": context_text},
            **({"llm_query_fn": llm_query_fn} if llm_query_fn else {}),
        },
    )

    prompt = [{"role": "user", "content": direct_prompt}]
    init_messages, info = env.init(prompt)

    print("=== Init Messages ===")
    for msg in init_messages:
        preview = msg["content"][:300].replace("\n", "\\n")
        print(f"  [{msg['role']}] {preview}...")

    # Turn 1: explore the context variable
    action_1 = (
        "Let me inspect the context.\n\n"
        "```python\nprint(context[:100])\n```"
    )
    step_out = env.step(action_1)
    print("\n=== Turn 1 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")
    for obs in step_out["observations"]:
        print(f"  [{obs['role']}] {obs['content']}")

    # Turn 2: test llm_query if available, otherwise go straight to answer
    if llm_query_fn:
        action_2 = (
            "Let me use llm_query to extract the answer.\n\n"
            '```python\nanswer = llm_query("What is the capital city mentioned in this text? '
            'Return only the city name.\\n\\n" + context)\n'
            'print(f"llm_query returned: {answer}")\n'
            "Final = answer.strip()\n```"
        )
    else:
        action_2 = (
            "I can see the answer.\n\n"
            '```python\nFinal = "Paris"\n```'
        )
    step_out = env.step(action_2)
    print("\n=== Turn 2 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")
    for obs in step_out["observations"]:
        print(f"  [{obs['role']}] {obs['content']}")

    print("\n=== Metrics ===")
    for k, v in env.get_metrics().items():
        print(f"  {k}: {v}")

    env.close()
