
import argparse
import json
from pathlib import Path
import hashlib

from datasets import load_dataset

SEED = 0

N_TARGET = {"refusal": 1000, "instruction-following": 1000, "coding": 1000}


# Exact Tülu `source` names, checked on 2026-11-16 with Counter(tulu["source"]).
TULU_SOURCES = {
    "coding": "ai2-adapt-dev/personahub_code_v2_34999",
    "instruction-following": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
    ,
}

# ---------------------------------------------------------------- rows and text
def row(id, prompt, response, task_type, is_refusal, source):
    return {
        "id": id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "task_type": task_type,
        "is_refusal_example": is_refusal,
        "source": source,
    }

def wildguard_rows():
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["response"] is not None and not x["adversarial"] 
                   and x["prompt_harm_label"]=='harmful' and x["response_refusal_label"]=='refusal')
    ds = ds.shuffle(seed=SEED).select(range(N_TARGET['refusal']))
    out = {"refusal": []}
    for x in ds:
        # WildGuardMix has no id column, so the id is a SHA-256 hash of the prompt.
        out["refusal"].append(row(hashlib.sha256(x["prompt"].encode("utf-8")).hexdigest()
                                  ,x["prompt"], x["response"], "safety", True, "wildguardmix"))
    return out


def tulu_rows():
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    out = {'coding':[], 'instruction-following':[]}
    for task, key in TULU_SOURCES.items():
        sub = ds.filter(lambda x: key in x["source"]
                        and "wildguard" not in x["source"]
                        and len(x["messages"]) == 2
                        and x["messages"][0]["role"] == "user")
        sub = sub.shuffle(seed=SEED).select(range(min(N_TARGET[task], len(sub))))
        for x in sub:
          # Keep Tülu's own id so each row can be traced back to the source record;
          # fall back to a prompt hash only if a record has no id.
          rid = x.get("id") or hashlib.sha256(x["messages"][0]['content'].encode("utf-8")).hexdigest()
          out[task].append(row(rid, x["messages"][0]['content'], x['messages'][1]['content'],
                               task, False, x["source"]))
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(out_dir="data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    wildguard = wildguard_rows()
    tulu = tulu_rows()
    write_jsonl(out_dir+'/wildguard.jsonl', wildguard['refusal'])
    write_jsonl(out_dir+'/tulu_coding.jsonl', tulu['coding'])
    write_jsonl(out_dir+'/tulu_instruction-following.jsonl', tulu['instruction-following'])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    main(ap.parse_args().out_dir)
