import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.qasper.utils import compute_all_metrics, compute_score
from skyrl_gym.tools.core import tool, ToolGroup


@dataclass
class QasperEnvConfig:
    max_turns: int = 10


class QasperToolGroup(ToolGroup):
    def __init__(self, paper_lines: List[Dict[str, Any]]):
        self.paper_lines = paper_lines
        super().__init__(name="QasperToolGroup")

    @tool
    def read(self, start: int, end: int) -> str:
        s = max(0, start)
        e = min(len(self.paper_lines) - 1, end)
        if s > e:
            return "(invalid range)"
        return "\n".join(f"{i}| {self.paper_lines[i]['text']}" for i in range(s, e + 1))

    @tool
    def search(self, query: str, context: int = 2, max_results: int = 15) -> str:
        if not query:
            return "(empty query)"

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        match_indices = [i for i, line in enumerate(self.paper_lines) if pattern.search(line["text"])]

        if not match_indices:
            return f'(no matches for "{query}")'

        groups: List[List[int]] = []
        for idx in match_indices:
            s = max(0, idx - context)
            e = min(len(self.paper_lines) - 1, idx + context)
            if groups and s <= groups[-1][1] + 1:
                groups[-1][1] = max(groups[-1][1], e)
            else:
                groups.append([s, e])

        shown = groups[:max_results]
        parts = []
        for gi, (s, e) in enumerate(shown):
            snippet = "\n".join(f"{i}| {self.paper_lines[i]['text']}" for i in range(s, e + 1))
            parts.append(f"--- match group {gi} (lines {s}-{e}) ---\n{snippet}")

        remaining = len(groups) - len(shown)
        if remaining > 0:
            parts.append(f"(+{remaining} more match groups)")

        return f"{len(match_indices)} matches found.\n\n" + "\n\n".join(parts)


class QasperEnv(BaseTextEnv):
    def __init__(self, env_config: Union[QasperEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"
        assert "extra_info" in extras, "extra_info field is required"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.paper_text = extras["extra_info"]["paper"]
        self.paper_lines = extras["extra_info"]["paper_lines"]
        self.max_turns = env_config.max_turns if hasattr(env_config, "max_turns") else 10

        self.tool_group = QasperToolGroup(paper_lines=self.paper_lines)
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []
        self._final_metrics: Optional[Dict[str, float]] = None

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0.0
        chat_history_str = "".join(item["content"] for item in self.chat_history)
        self._final_metrics = compute_all_metrics(
            chat_history_str, self.ground_truth, self.paper_text, self.paper_lines
        )
        return self._final_metrics["f1"]

    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[List]]:
        if "<search>" in action and "</search>" in action:
            m = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
            if m:
                return "search", [m.group(1).strip()]

        if "<read>" in action and "</read>" in action:
            m = re.search(r"<read>(.*?)</read>", action, re.DOTALL)
            if m:
                parts = m.group(1).strip().split(",")
                if len(parts) == 2:
                    try:
                        return "read", [int(parts[0].strip()), int(parts[1].strip())]
                    except ValueError:
                        pass

        return None, None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        tool_name, tool_args = self._parse_action(action)
        observation = None
        error = None
        info = {}

        if tool_name and tool_args is not None:
            try:
                raw = self._execute_tool("QasperToolGroup", tool_name, tool_args)
                observation = "\n<information>" + raw + "</information>\n"
                info = {"tool_name": tool_name, "tool_input": tool_args}
            except Exception as e:
                error = str(e)
        else:
            error = "No valid tool call found. Use <search>query</search> or <read>start,end</read>."

        new_obs = {"role": "user", "content": observation if observation else error}
        self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=done,
            metadata=info,
        )

    def get_metrics(self) -> Dict[str, Any]:
        if self._final_metrics is not None:
            return self._final_metrics
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
