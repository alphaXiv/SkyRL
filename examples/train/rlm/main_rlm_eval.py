"""Eval-only entry point for the Recursive Language Model (RLM) environment.

Mirrors ``skyrl.train.entrypoints.main_generate`` but uses ``RLMConfig`` and
constructs ``RLMGymGenerator`` via an overridden ``get_generator`` so the
RLM-specific hooks fire during eval rollouts too.
"""

import asyncio
import sys
import time

import ray
from loguru import logger

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg
from .openrouter_client import OpenRouterInferenceClient
from .rlm_config import RLMGeneratorConfig
from .rlm_generator import RLMGymGenerator


RLMConfig = make_config(generator_cls=RLMGeneratorConfig)


class RLMEvalEntrypoint(EvalOnlyEntrypoint):
    def get_colocate_pg(self, **kwargs):
        # Eval-only: no GPU colocation needed.
        return None

    def get_inference_client(self):
        model = getattr(self.cfg.generator, "frozen_openrouter_model", None)
        if model:
            logger.info(f"Using OpenRouter inference client with model: {model}")
            return OpenRouterInferenceClient.from_model(model=model, tokenizer=self.tokenizer)
        return super().get_inference_client()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return RLMGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg) -> dict:
    exp = RLMEvalEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = RLMConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    t0 = time.time()
    metrics = ray.get(eval_entrypoint.remote(cfg))
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("EVAL RESULTS")
    logger.info("=" * 60)
    priority_keys = ["eval/all/avg_score", "eval/all/mean_positive_reward"]
    pass_at_keys = sorted(k for k in metrics if "pass_at" in k)
    other_keys = sorted(k for k in metrics if k not in priority_keys and k not in pass_at_keys)
    for k in priority_keys + pass_at_keys + other_keys:
        if k in metrics:
            v = metrics[k]
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info(f"  total_eval_time_s: {elapsed:.1f}")
    batch_size = cfg.trainer.eval_batch_size
    if batch_size:
        logger.info(f"  avg_time_per_sample_s: {elapsed / batch_size:.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
