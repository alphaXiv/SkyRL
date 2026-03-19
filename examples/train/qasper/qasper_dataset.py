# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QASPER dataset to parquet format
"""

import argparse
import json
import os
import re

import datasets


def split_into_sentences(text: str) -> list:
    boundaries = []
    for m in re.finditer(r"[.!?](?=\s|$)", text):
        boundaries.append(m.start() + len(m.group()))
    if not boundaries or boundaries[-1] < len(text):
        boundaries.append(len(text))

    lines = []
    pos = 0
    for boundary in boundaries:
        chunk = text[pos:boundary]
        stripped = chunk.strip()
        if stripped:
            leading = len(chunk) - len(chunk.lstrip())
            char_start = pos + leading
            char_end = char_start + len(stripped)
            lines.append({"char_start": char_start, "char_end": char_end, "text": stripped})
        pos = boundary
    return lines


SYSTEM_TEMPLATE = """\
You are an evidence retrieval agent. Given a question about an academic paper, your job is to identify the line ranges that contain evidence for answering the question.

Paper: {title}

Abstract: {abstract}

The paper has been split into {num_lines} numbered lines (0 to {max_line}). You have two tools:
- <search>query</search> — search for a keyword or phrase; returns matching lines with surrounding context and line numbers
- <read>start,end</read> — read a range of lines by start and end line number (inclusive, 0-indexed)

Use these tools to explore the paper thoroughly, then output your final answer as:
<answer>[a,b] [c,d] ...</answer>

where each [a,b] is an inclusive line range. Merge adjacent or overlapping ranges.\
"""

INSTRUCTION_SUFFIX = "Rapidly retrieve the relevant line numbers in a list like [a,b] [c,d]."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="~/projects/SkyRL/examples/train/qasper/raw_json/test.json")
    parser.add_argument("--output_dir", default="~/data/qasper")

    args = parser.parse_args()

    args.input_path = os.path.expanduser(args.input_path)
    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "qasper"

    with open(args.input_path) as f:
        raw_data = json.load(f)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            evidence = example["evidence"]
            paper = "\n\n".join(example["paragraphs"])
            paper_lines = split_into_sentences(paper)
            num_lines = len(paper_lines)

            system_message = SYSTEM_TEMPLATE.format(
                title=example["title"],
                abstract=example["abstract"],
                num_lines=num_lines,
                max_line=num_lines - 1,
            )

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": question_raw + " " + INSTRUCTION_SUFFIX,
                    },
                ],
                "env_class": "qasper",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": evidence,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": evidence,
                    "question": question_raw,
                    "paper": paper,
                    "paper_lines": paper_lines,
                    "title": example["title"],
                    "abstract": example["abstract"],
                },
            }
            return data

        return process_fn

    dataset = datasets.Dataset.from_list(raw_data)
    dataset = dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))
