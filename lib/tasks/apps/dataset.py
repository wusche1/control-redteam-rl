import json
from huggingface_hub import hf_hub_download


def load_apps(split: str, difficulty: str = "interview", max_samples: int | None = None) -> list[dict]:
    path = hf_hub_download("codeparrot/apps", f"{split}.jsonl", repo_type="dataset")
    results = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("difficulty") != difficulty:
                continue
            try:
                test_cases = json.loads(row["input_output"])
            except (json.JSONDecodeError, TypeError):
                continue
            try:
                solutions = json.loads(row["solutions"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not solutions:
                continue
            inputs = test_cases.get("inputs", [])
            outputs = test_cases.get("outputs", [])
            if not inputs or not outputs:
                continue
            results.append({
                "problem_id": str(row["id"]),
                "question": row["question"],
                "test_cases": [{"input": i, "output": o} for i, o in zip(inputs, outputs)],
                "solutions": solutions,
            })
            if max_samples and len(results) >= max_samples:
                break
    return results
