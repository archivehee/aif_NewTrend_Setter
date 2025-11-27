import os
import json
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = BASE_DIR
os.makedirs(OUTPUT_ROOT, exist_ok=True)

HOME_DIR = os.path.expanduser("~")
DEFAULT_ITEM2ID_ROOT = os.path.join(HOME_DIR, "aigs", "NTS", "data")
DEFAULT_ITEM_META_ROOT = os.path.join(
    HOME_DIR, "edda_backbone", "preprocess_raw", "amazon", "23", "item_meta"
)

ENV_ITEM2ID_ROOT = os.environ.get("ITEM2ID_ROOT", DEFAULT_ITEM2ID_ROOT)
ENV_ITEM_META_ROOT = os.environ.get("ITEM_META_ROOT", DEFAULT_ITEM_META_ROOT)

REQUIRED_FIELDS = ["main_category", "categories", "title", "description"]

DOMAIN_SHORT_NAME = {
    "Home_and_Kitchen": "home",
}

EMB_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 64
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY")
NOMIC_ENDPOINT = "https://api-atlas.nomic.ai/v1/embedding/text"


def request_embeddings(texts: List[str]) -> np.ndarray:
    if NOMIC_API_KEY is None:
        raise RuntimeError("need NOMIC_API_KEY")
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMB_MODEL,
        "texts": texts,
        "dimensionality": EMBEDDING_DIM,
    }

    resp = requests.post(NOMIC_ENDPOINT, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(
            f"calling Nomic embedding API fail: status={resp.status_code}, body={resp.text}"
        )

    data = resp.json()
    emb = np.array(data["embeddings"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding shape {emb.shape} (expected (?, {EMBEDDING_DIM}))"
        )
    if emb.shape[0] != len(texts):
        raise ValueError(
            f"Mismatched embedding count {emb.shape[0]} for {len(texts)} texts"
        )
    return emb


def load_item2id(path: str):
    idx2item: List[str] = []
    item2idx: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_iid, idx_str = line.split()
            idx = int(idx_str)

            if idx >= len(idx2item):
                idx2item.extend([None] * (idx - len(idx2item) + 1))
            idx2item[idx] = raw_iid
            item2idx[raw_iid] = idx

    if any(i is None for i in idx2item):
        raise ValueError(f"Some item indices are missing in {path}")

    return idx2item, item2idx


def normalize_field_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        parts = [normalize_field_value(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None
    return str(value)


def load_item_meta(path: str) -> Dict[str, Dict[str, Optional[str]]]:
    meta: Dict[str, Dict[str, Optional[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            parent_asin = record.get("parent_asin")
            if parent_asin is None:
                continue

            field_values = {
                field: normalize_field_value(record.get(field))
                for field in REQUIRED_FIELDS
            }
            meta[parent_asin] = field_values

    return meta


def build_field_texts(idx2item: List[str], meta: Dict[str, Dict[str, Optional[str]]]):
    field_texts: Dict[str, List[Optional[str]]] = {
        field: [] for field in REQUIRED_FIELDS
    }
    for item_id in idx2item:
        entry = meta.get(item_id, {})
        for field in REQUIRED_FIELDS:
            field_texts[field].append(entry.get(field))
    return field_texts


def embed_field_texts(field: str, texts: List[Optional[str]]) -> np.ndarray:
    embs = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    missing_count = 0
    batch_texts: List[str] = []
    batch_indices: List[int] = []

    with tqdm(total=len(texts), desc=f"Embedding {field}", leave=False) as pbar:
        for idx, text in enumerate(texts):
            if text is None:
                missing_count += 1
            else:
                batch_indices.append(idx)
                batch_texts.append(text)
                if len(batch_texts) >= EMBEDDING_BATCH_SIZE:
                    batch_embs = request_embeddings(batch_texts)
                    embs[batch_indices] = batch_embs
                    batch_texts.clear()
                    batch_indices.clear()
            pbar.update(1)

    if batch_texts:
        batch_embs = request_embeddings(batch_texts)
        embs[batch_indices] = batch_embs

    print(f"[{field}] zero-fill count: {missing_count}/{len(texts)}")
    return embs


def build_domain_config(item2id_root: str, item_meta_root: str, cold: bool):
    """
    if cold=True, "maps_c", else "maps" 
    """
    config = {}

    for domain in ["Home_and_Kitchen"]: # if other domain exist
        maps_dir = "maps_c" if cold else "maps"
        item2id_path = os.path.join(item2id_root, domain, maps_dir, "item2id.txt")
        meta_path = os.path.join(item_meta_root, f"meta_{domain}.jsonl")
        config[domain] = {
            "item2id": item2id_path,
            "metadata": meta_path,
        }
    return config


def process_domain(domain: str, config: Dict[str, Dict[str, str]], cold: bool):
    print(f"\nDomain: {domain} (cold={cold})")
    dconf = config[domain]

    idx2item, item2idx = load_item2id(dconf["item2id"])
    print(f"#items: {len(idx2item)}")

    item_meta = load_item_meta(dconf["metadata"])
    print(f"#meta entries: {len(item_meta)}")

    field_texts = build_field_texts(idx2item, item_meta)

    field_embs = OrderedDict()
    for field in REQUIRED_FIELDS:
        field_embs[field] = embed_field_texts(field, field_texts[field])
        print(f"[{field}] emb shape: {field_embs[field].shape}")

    short_name = DOMAIN_SHORT_NAME.get(domain, domain.lower())
    suffix = "_c" if cold else ""
    out_fname = f"itm_txt_emb_{short_name}{suffix}.pkl"
    out_path = os.path.join(OUTPUT_ROOT, out_fname)

    save_dict = OrderedDict()
    save_dict["field_emb"] = field_embs
    save_dict["idx2item"] = idx2item
    save_dict["item2idx"] = item2idx
    save_dict["meta"] = {
        "domain": domain,
        "model": EMB_MODEL,
        "dim": EMBEDDING_DIM,
        "fields": REQUIRED_FIELDS,
        "item2id_path": dconf["item2id"],
        "item_meta_path": dconf["metadata"],
        "cold_split": cold,
        "note": "when cold=True, aligned with *_c train/valid/test (maps_c, train_c/valid_c/test_c).",
    }

    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        default="Home_and_Kitchen",
        choices=["Home_and_Kitchen"],  
        help="target domain",
    )
    parser.add_argument(
        "--item2id-root",
        type=str,
        default=ENV_ITEM2ID_ROOT,
        help="root path of item2id (default: $ITEM2ID_ROOT or ~/aigs/NTS/data)",
    )
    parser.add_argument(
        "--item-meta-root",
        type=str,
        default=ENV_ITEM_META_ROOT,
        help="root path of item_meta (default: $ITEM_META_ROOT or edda_backbone/.../item_meta)",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="if set, use maps_c, align with train_c/valid_c/test_c, and output *_c.pkl",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # domain config gen (maps vs maps_c)
    domain_config = build_domain_config(
        item2id_root=args.item2id_root,
        item_meta_root=args.item_meta_root,
        cold=args.cold,
    )

    process_domain(args.domain, domain_config, cold=args.cold)


if __name__ == "__main__":
    main()

'''
default
python itm_emb.py \
  --domain Home_and_Kitchen
      
-------------------------------------------
python itm_emb.py \
  --domain Home_and_Kitchen \
  --cold 
cold option -> load "train/valid/test_c.txt"
'''
