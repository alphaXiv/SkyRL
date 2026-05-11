"""``EvidenceRLMEnv``: an RLM environment for evidence-extraction tasks.

Subclasses ``BaseRLMEnv`` and supplies:
- evidence-F1 reward (single-paper or multi-paper) built from ``reward_spec.evidence``
- paper-aware REPL tools (``list_papers``, ``search``, ``extract_section``, ``get_paper_abstract``)
- multipaper parent / child system prompts

The subclass overrides ``_get_reward`` with an LLM-judge scorer.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

from skyrl_gym.envs.rlm.env import BaseRLMEnv
from skyrl_gym.metrics import default_aggregate_metrics

from .evidence_rewards import judge_reward
from .paper_tools import make_tools


# ---------------------------------------------------------------------------
# Multipaper system prompts (parent + child)
# ---------------------------------------------------------------------------

MULTIPAPER_PARENT_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an evidence extraction coordinator that finds VERBATIM text relevant to a query across a collection of papers. `context` is a dict mapping paper IDs to full paper texts.

REPL tools:
- `context`: dict mapping paper IDs to full paper texts.
- `list_papers(context)` — list all paper IDs with content previews.
- `search(text, keyword, window=300)` — keyword search. Pass `context` to search all papers, or `context[paper_id]` for one.
- `get_paper_abstract(context, paper_id)` — return the paper's title and abstract.
- `rlm_query_batched(prompts, context_list=None)` — dispatch child agents (max 4 per call).
- `FINAL_VAR(variable_name)` — return your final answer (a list of verbatim substrings).\
"""
)

MULTIPAPER_CHILD_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are a precise evidence extraction worker. You have a single paper in `context` and a query. Return ALL verbatim passages that directly answer the query.

REPL tools:
- `context`: full text of your paper.
- `search(text, keyword, window=300, bidirectional=True)` — keyword search; returns a list of snippets.
- `extract_section(snippet, start_phrase, end_phrase)` — extract a substring between two phrases (inclusive, case-insensitive).
- `FINAL_VAR(variable_name)` — return your final answer (a list of verbatim substrings).\
"""
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_paper_texts(extras: Dict[str, Any]) -> Dict[str, str]:
    """Extract the {paperId: text} dict from extra_info.context_text, or empty dict.

    Plain-string context (single-paper datasets) is wrapped under the key
    ``"__single__"`` so the verbatim short-circuit in ``judge_reward`` fires.
    """
    raw = (extras.get("extra_info") or {}).get("context_text")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    if isinstance(raw, str):
        return {"__single__": raw}
    return {}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EvidenceRLMEnv(BaseRLMEnv):
    """RLM environment for single-paper evidence extraction.

    Acts as the worker role: one paper in ``context``, returns a list of
    verbatim evidence spans. Uses ``MULTIPAPER_CHILD_SYSTEM_PROMPT``.

    Reward: LLM-judge precision/recall over predicted vs. ground-truth
    evidence from ``extras["reward_spec"]["evidence"]``.
    """

    SYSTEM_PROMPT = MULTIPAPER_CHILD_SYSTEM_PROMPT
    JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
    JUDGE_BASE_URL = "https://api.openai.com/v1"

    def _get_reward(self, final_answer: str) -> float:
        evidence = (self.extras.get("reward_spec") or {}).get("evidence") or []
        paper_texts = _extract_paper_texts(self.extras)
        reward, precision, recall, extras = judge_reward(
            final_answer,
            question=self._root_prompt,
            evidence=evidence,
            model=self.JUDGE_MODEL,
            base_url=self.JUDGE_BASE_URL,
            paper_texts=paper_texts,
        )
        self._judge_precision = precision
        self._judge_recall = recall
        self._judge_per_paper = extras.get("per_paper", {})
        self._predicted_paper_ids = extras.get("predicted_paper_ids", [])
        return reward

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["depth"] = self.extras.get("depth", 0)
        if hasattr(self, "_judge_precision") and hasattr(self, "_judge_recall"):
            metrics["judge_precision"] = self._judge_precision
            metrics["judge_recall"] = self._judge_recall
        return metrics

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_repl_tools(self) -> Dict[str, Any]:
        return make_tools()


class MultipaperEvidenceRLMEnv(EvidenceRLMEnv):
    """Multi-paper evidence extraction with parent/child orchestration.

    Root agent (depth 0) gets ``MULTIPAPER_PARENT_SYSTEM_PROMPT`` and
    coordinates: it picks relevant papers and dispatches child agents via
    ``rlm_query_batched``. Each child rollout (depth >= 1) runs as a
    worker with ``MULTIPAPER_CHILD_SYSTEM_PROMPT`` over a single paper.

    The generator stamps ``extras["depth"]`` per rollout; this class reads
    it to pick the right prompt.
    """
    def _get_reward(self, final_answer: str) -> float:
        depth = self.extras.get("depth", 0)
        if depth > 0: return 0.0 # short circuit child rewards to 0

        evidence = (self.extras.get("reward_spec") or {}).get("evidence") or []
        paper_texts = _extract_paper_texts(self.extras)
        reward, precision, recall, extras = judge_reward(
            final_answer,
            question=self._root_prompt,
            evidence=evidence,
            model=self.JUDGE_MODEL,
            base_url=self.JUDGE_BASE_URL,
            paper_texts=paper_texts,
        )
        self._judge_precision = precision
        self._judge_recall = recall
        self._judge_per_paper = extras.get("per_paper", {})
        self._predicted_paper_ids = extras.get("predicted_paper_ids", [])
        return reward

    def _get_system_prompt(self) -> str:
        depth = self.extras.get("depth", 0)
        return MULTIPAPER_PARENT_SYSTEM_PROMPT if depth == 0 else MULTIPAPER_CHILD_SYSTEM_PROMPT

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        depth = self.extras.get("depth", 0)
        if depth == 0:
            metrics["query"] = self._root_prompt
            evidence = (self.extras.get("reward_spec") or {}).get("evidence") or []
            metrics["ground_truth_paper_ids"] = [e["paperId"] for e in evidence if "paperId" in e]
            metrics["ground_truth_evidence"] = evidence
            metrics["context_paper_ids"] = list(_extract_paper_texts(self.extras).keys())
            if hasattr(self, "_predicted_paper_ids"):
                metrics["predicted_paper_ids"] = self._predicted_paper_ids
            if hasattr(self, "_judge_per_paper"):
                metrics["judge_per_paper"] = self._judge_per_paper
        return metrics

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split rollouts by depth: depth=0 → parent/*, depth>=1 → child/*."""
        parents = [m for m in metrics if m.get("depth", 0) == 0]
        children = [m for m in metrics if m.get("depth", 0) > 0]
        out: Dict[str, Any] = {}
        out.update({f"parent/{k}": v for k, v in default_aggregate_metrics(parents).items()})
        out.update({f"child/{k}": v for k, v in default_aggregate_metrics(children).items()})
        return out
