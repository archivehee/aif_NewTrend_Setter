import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DOMAIN = "Home_and_Kitchen"
TARGET_USERS = 5000
SRC = Path("/home/heek/edda_backbone/preprocess_raw/amazon/23/user_reviews/5_core") / f"{DOMAIN}.json"

DATA_ROOT = Path("/home/heek/aigs/NTS/data") / DOMAIN
MAP_DIR = DATA_ROOT / "maps_c"
MANIFEST_PATH = DATA_ROOT / "manifest.json"


def get_timestamp(record: dict) -> int:
    """Return millisecond timestamp from record."""
    if "timestamp" in record:
        return int(record["timestamp"])
    return int(record.get("unixReviewTime", 0))


def count_item_frequency() -> Counter:
    """Count how many times each item appears in the source file."""
    freq = Counter()
    with SRC.open("r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            item = data.get("parent_asin") or data.get("asin")
            if item:
                freq[item] += 1
    return freq


def classify_items(item_freq: Counter):
    """Return helper closure that maps an item frequency to a popularity group."""
    counts = sorted(item_freq.values())
    n_items = len(counts)
    thr1 = counts[n_items // 3]
    thr2 = counts[(2 * n_items) // 3]

    def group_of(freq: int) -> int:
        if freq <= thr1:
            return 0
        if freq <= thr2:
            return 1
        return 2

    return group_of


def choose_candidate_users(item_freq: Counter):
    """Select users that have interactions in all popularity groups."""
    group_of = classify_items(item_freq)
    user_groups = defaultdict(lambda: [0, 0, 0])
    with SRC.open("r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            user = data.get("user_id")
            item = data.get("parent_asin") or data.get("asin")
            if not user or not item or item not in item_freq:
                continue
            g = group_of(item_freq[item])
            user_groups[user][g] += 1

    def balance_key(counts):
        mean = sum(counts) / 3.0
        var = sum((c - mean) ** 2 for c in counts) / 3.0
        std = math.sqrt(var)
        # prefer users with more interactions as a tiebreaker
        return (std, -sum(counts))

    valid_users = [(u, counts) for u, counts in user_groups.items() if all(c > 0 for c in counts)]
    valid_users.sort(key=lambda x: balance_key(x[1]))
    return [u for u, _ in valid_users]


def collect_interactions(valid_users):
    """Gather chronological interaction list for each candidate user."""
    interactions = {u: [] for u in valid_users}
    unique_items = set()
    user_set = set(valid_users)
    with SRC.open("r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            user = data.get("user_id")
            if user not in user_set:
                continue
            item = data.get("parent_asin") or data.get("asin")
            ts = get_timestamp(data)
            if not item or ts <= 0:
                continue
            unique_items.add(item)
            interactions[user].append((ts, item))
    return interactions, unique_items


def select_users_with_history(valid_users, interactions):
    """Pick TARGET_USERS users that have at least two interactions."""
    selected = []
    for user in valid_users:
        if len(interactions[user]) < 2:
            continue
        selected.append(user)
        if len(selected) == TARGET_USERS:
            break
    if len(selected) < TARGET_USERS:
        raise RuntimeError(f"Only found {len(selected)} users with >=2 interactions")
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

    item_freq = count_item_frequency()
    candidate_users = choose_candidate_users(item_freq)
    interactions, unique_items = collect_interactions(candidate_users)
    selected_users = select_users_with_history(candidate_users, interactions)

    user2id = {user: idx for idx, user in enumerate(selected_users)}

    selected_items = set()
    for user in selected_users:
        for _, item in interactions[user]:
            selected_items.add(item)
    item2id = {item: idx for idx, item in enumerate(sorted(selected_items))}
    train_pairs, valid_pairs, test_pairs = [], [], []

    for user in selected_users:
        history = sorted(interactions[user], key=lambda x: x[0])
        if len(history) < 2:
            continue  # should not happen, but guard anyway
        mapped_items = [item2id[item] for _, item in history if item in item2id]
        if len(mapped_items) < 2:
            continue
        uid = user2id[user]
        test_item = mapped_items[-1]
        valid_item = mapped_items[-2]
        for iid in mapped_items[:-2]:
            train_pairs.append((uid, iid))
        valid_pairs.append((uid, valid_item))
        test_pairs.append((uid, test_item))

    train_pairs.sort()
    valid_pairs.sort()
    test_pairs.sort()

    dump_pairs(DATA_ROOT / "train_c.txt", train_pairs)
    dump_pairs(DATA_ROOT / "valid_c.txt", valid_pairs)
    dump_pairs(DATA_ROOT / "test_c.txt", test_pairs)
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
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)

    print(f"Done. users={len(user2id)} items={len(item2id)}")
    print("train/valid/test sizes:", len(train_pairs), len(valid_pairs), len(test_pairs))


if __name__ == "__main__":
    main()