import os
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

DOMAIN = "Home_and_Kitchen"
TARGET_USERS = 5000
MIN_INTERACTIONS = 5  # after meta filtering

BASE_DIR = Path("/home/heek/edda_backbone/preprocess_raw/amazon/23")
REVIEW_DIR = BASE_DIR / "user_reviews/5_core"
META_DIR = BASE_DIR / "item_meta"

DATA_ROOT = Path("/home/heek/aigs/NTS/data") / DOMAIN
MAP_DIR = DATA_ROOT / "maps"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

REQUIRED_FIELDS = ["main_category", "categories", "title", "description"]


def get_timestamp(record: dict) -> int:
    """Return millisecond timestamp from record."""
    if "timestamp" in record:
        return int(record["timestamp"])
    return int(record.get("unixReviewTime", 0))


def is_valid_field(value):
    if value in [None, "", [], {}, "null"]:
        return False

    s = str(value).strip().lower()
    if s in ["", "[]", "{}", "none", "null", "[ ]"]:
        return False

    # string that looks like a list
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list) and all(str(v).strip().lower() in ["", "none", "null"] for v in parsed):
                return False
        except Exception:
            pass

    # real list
    if isinstance(value, list) and all(str(v).strip().lower() in ["", "none", "null"] for v in value):
        return False

    return True


def load_valid_items() -> set:
    """Return set of asins whose meta fields are all valid."""
    meta_path = META_DIR / f"meta_{DOMAIN}.jsonl"
    valid_asins = set()

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            asin = entry.get("parent_asin") or entry.get("asin")
            if not asin:
                continue
            if all(is_valid_field(entry.get(field)) for field in REQUIRED_FIELDS):
                valid_asins.add(asin)

    print(f"[{DOMAIN}] valid meta items: {len(valid_asins)}")
    return valid_asins


def collect_user_interactions(valid_items: set):
    """
    Collect interactions (ts, item) per user, only for items in valid_items.
    Also return item frequency over filtered interactions.
    """
    review_path = REVIEW_DIR / f"{DOMAIN}.json"
    interactions = defaultdict(list)
    item_freq = Counter()

    with review_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue
            user = data.get("user_id")
            item = data.get("parent_asin") or data.get("asin")
            if not user or not item or item not in valid_items:
                continue
            ts = get_timestamp(data)
            if ts <= 0:
                continue
            interactions[user].append((ts, item))
            item_freq[item] += 1

    print(f"[{DOMAIN}] users with >=1 valid interaction: {len(interactions)}")
    return interactions, item_freq


def choose_users(interactions: dict):
    """Select TARGET_USERS users with at least MIN_INTERACTIONS interactions."""
    candidates = [u for u, hist in interactions.items() if len(hist) >= MIN_INTERACTIONS]

    if len(candidates) < TARGET_USERS:
        raise RuntimeError(
            f"Not enough users with >= {MIN_INTERACTIONS} interactions: "
            f"required {TARGET_USERS}, available {len(candidates)}"
        )

    random.seed(2025)
    selected = random.sample(candidates, TARGET_USERS)
    print(f"[{DOMAIN}] selected users: {len(selected)}")
    return selected


def dump_pairs(path: Path, pairs):
    with path.open("w", encoding="utf-8") as fout:
        for u, i in pairs:
            fout.write(f"{u} {i}\n")


def dump_mapping(path: Path, mapping: dict):
    with path.open("w", encoding="utf-8") as fout:
        for key, idx in mapping.items():
            fout.write(f"{key} {idx}\n")

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. valid items by meta
    valid_items = load_valid_items()

    # 2. interactions filtered by valid items
    interactions_all, item_freq = collect_user_interactions(valid_items)

    # 3. choose users
    selected_users = choose_users(interactions_all)

    # 4. gather items for selected users
    selected_items = set()
    for u in selected_users:
        for ts, item in interactions_all[u]:
            selected_items.add(item)
    item2id = {item: idx for idx, item in enumerate(sorted(selected_items))}
    user2id = {user: idx for idx, user in enumerate(selected_users)}

    print(f"[{DOMAIN}] items: {len(item2id)}, users: {len(user2id)}")

    # 5. build train/valid/test with leave-one-out
    train_pairs = []
    eval_candidates = []  # (uid, valid_iid, test_iid)

    for user in selected_users:
        hist = sorted(interactions_all[user], key=lambda x: x[0])
        mapped_items = [item2id[item] for _, item in hist if item in item2id]
        if len(mapped_items) < 3:
            # should be rare due to MIN_INTERACTIONS, but keep guard
            continue
        uid = user2id[user]
        test_item = mapped_items[-1]
        valid_item = mapped_items[-2]

        # train은 항상 앞의 것들
        for iid in mapped_items[:-2]:
            train_pairs.append((uid, iid))

        # valid/test는 일단 후보로만 저장
        eval_candidates.append((uid, valid_item, test_item))

    # 6. global train item set 기준으로 cold 제거
    train_pairs.sort()
    train_items = {iid for _, iid in train_pairs}

    valid_pairs, test_pairs = [], []
    for uid, v_iid, t_iid in eval_candidates:
        if v_iid in train_items:
            valid_pairs.append((uid, v_iid))
        if t_iid in train_items:
            test_pairs.append((uid, t_iid))

    valid_pairs.sort()
    test_pairs.sort()

    # cold ratio 계산 (이제 이론상 0이어야 함)
    cold_valid = sum(1 for _, iid in valid_pairs if iid not in train_items)
    cold_test = sum(1 for _, iid in test_pairs if iid not in train_items)
    total_eval = len(valid_pairs) + len(test_pairs)
    cold_ratio = (cold_valid + cold_test) / total_eval if total_eval > 0 else 0.0

    print(f"Done. users={len(user2id)} items={len(item2id)}")
    print("train/valid/test sizes:", len(train_pairs), len(valid_pairs), len(test_pairs))
    print(f"eval cold pairs: {cold_valid + cold_test}/{total_eval} ({cold_ratio:.4f})")

    # 7. dump
    dump_pairs(DATA_ROOT / "train.txt", train_pairs)
    dump_pairs(DATA_ROOT / "valid.txt", valid_pairs)
    dump_pairs(DATA_ROOT / "test.txt", test_pairs)
    dump_mapping(MAP_DIR / "user2id.txt", user2id)
    dump_mapping(MAP_DIR / "item2id.txt", item2id)

    manifest = {
        "users": len(user2id),
        "items_per_domain": {DOMAIN: len(item2id)},
        "splits": {
            DOMAIN: {
                "train": len(train_pairs),
                "valid": len(valid_pairs),
                "test": len(test_pairs),
            }
        },
        "meta_filtered": {
            "required_fields": REQUIRED_FIELDS,
            "min_interactions": MIN_INTERACTIONS,
            "target_users": TARGET_USERS,
            "eval_cold_removed": True,
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)


if __name__ == "__main__":
    main()
