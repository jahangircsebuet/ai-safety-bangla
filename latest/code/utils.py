import json
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional
from itertools import combinations

FIELDS_TO_MOVE = [
    "response_bn",
    "aegis_category",
    "aegis_reason",
    "aegis_category_name",
    "prompt_safety",
    "harm_score",
]


import json
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set, Tuple, Optional

import unicodedata

import json, random, math
from collections import defaultdict
from typing import Dict, List, Any, Tuple

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def stratified_sample_aegis(
    data: List[dict],
    *,
    category_field: str = "most_severe_category",
    target_n: int = 300,
    min_per_category: int = 1,
    max_per_category: int = None,  # e.g., 80
    seed: int = 42,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Stratified sampling proportional to category frequency.
    - Uses Hamilton / largest-remainder rounding to hit exact target_n.
    - Respects min_per_category and optional max_per_category.
    """
    rng = random.Random(seed)

    # group items by category
    groups: Dict[str, List[dict]] = defaultdict(list)
    for obj in data:
        if not isinstance(obj, dict):
            continue
        cat = obj.get(category_field)
        cat = str(cat).strip() if cat is not None else ""
        if not cat:
            cat = "<missing>"
        groups[cat].append(obj)

    cats = list(groups.keys())
    total = sum(len(v) for v in groups.values())
    if total == 0:
        return [], {}

    # initial proportional quotas
    raw_quota = {c: (len(groups[c]) / total) * target_n for c in cats}

    # floor + remainder method to ensure exact target_n
    quota = {c: int(math.floor(raw_quota[c])) for c in cats}

    # apply minimums (but cannot exceed available)
    for c in cats:
        quota[c] = max(quota[c], min_per_category)
        quota[c] = min(quota[c], len(groups[c]))

    # apply max cap if given
    if max_per_category is not None:
        for c in cats:
            quota[c] = min(quota[c], max_per_category, len(groups[c]))

    # adjust to hit target_n exactly
    current = sum(quota.values())

    # helper to compute "room" for increasing a category
    def can_add(c: str) -> bool:
        if max_per_category is not None and quota[c] >= max_per_category:
            return False
        return quota[c] < len(groups[c])

    # If we have too many, remove from largest quotas first (but keep >= min_per_category if possible)
    if current > target_n:
        # sort by quota desc then remainder asc (remove from biggest first)
        order = sorted(cats, key=lambda c: (quota[c], raw_quota[c] - math.floor(raw_quota[c])), reverse=True)
        i = 0
        while current > target_n and i < 10_000:
            c = order[i % len(order)]
            if quota[c] > 0 and quota[c] > min_per_category:
                quota[c] -= 1
                current -= 1
            i += 1

    # If we have too few, add using largest remainders
    elif current < target_n:
        remainders = sorted(
            cats,
            key=lambda c: (raw_quota[c] - math.floor(raw_quota[c])),
            reverse=True,
        )
        i = 0
        while current < target_n and i < 10_000:
            c = remainders[i % len(remainders)]
            if can_add(c):
                quota[c] += 1
                current += 1
            i += 1

        # If still short (e.g., due to caps), fill from any category with remaining capacity
        if current < target_n:
            fill_cats = [c for c in cats if can_add(c)]
            j = 0
            while current < target_n and fill_cats and j < 1_000_000:
                c = fill_cats[j % len(fill_cats)]
                if can_add(c):
                    quota[c] += 1
                    current += 1
                j += 1

    # sample per category
    sampled: List[dict] = []
    per_cat_selected: Dict[str, int] = {}
    for c in cats:
        k = quota[c]
        rng.shuffle(groups[c])
        take = groups[c][:k]
        sampled.extend(take)
        per_cat_selected[c] = len(take)

    rng.shuffle(sampled)
    return sampled, per_cat_selected


def split_categories(x: Any) -> List[str]:
    """
    Handles:
      - string: "A" or "A, B"
      - list: ["A", "B"]
      - null/None
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def count_by_field(data: List[dict], field: str, multi: bool = False) -> Counter:
    cnt = Counter()
    for obj in data:
        if not isinstance(obj, dict):
            continue
        val = obj.get(field)
        if multi:
            for c in split_categories(val):
                cnt[c] += 1
        else:
            key = str(val).strip() if val is not None else ""
            if key:
                cnt[key] += 1
            else:
                cnt["<missing>"] += 1
    return cnt


def _norm_text(t: str, normalize_whitespace: bool = True, unicode_form: str = "NFC") -> str:
    """
    Normalize Bangla/Unicode text so visually identical strings match in code.

    - unicode_form: "NFC" is usually best for stable hashing/equality
      (you can switch to "NFKC" if you also want compatibility folding)
    - normalize_whitespace: collapses multiple spaces + trims ends
    """
    if t is None:
        return ""

    # Ensure string
    if not isinstance(t, str):
        t = str(t)

    # Unicode normalize (fixes composed vs decomposed variants)
    t = unicodedata.normalize(unicode_form, t)

    # Whitespace normalize (optional)
    if normalize_whitespace:
        t = " ".join(t.split())

    return t


def collect_texts_to_delete(dups_report_path: str) -> Set[str]:
    """
    From the duplicates report, collect text values from rows[1:] for each duplicate id.
    These are the ones we want to delete from the original dataset.
    """
    report = load_json(dups_report_path)
    to_delete: Set[str] = set()

    for d in report.get("duplicates", []):
        rows = d.get("rows", [])
        # keep the first one, delete the rest
        for r in rows[1:]:
            t = r.get("text")
            if isinstance(t, str):
                to_delete.add(t)

    return to_delete

def delete_items_by_text(
    dataset_path: str,
    dups_report_path: str,
    out_path: str,
    *,
    use_strip_match: bool = False
) -> Dict[str, Any]:
    """
    Deletes objects from dataset_path whose obj['text'] matches the duplicate texts (rows[1:]).
    Writes cleaned dataset to out_path.
    """
    texts_to_delete = collect_texts_to_delete(dups_report_path)
    data = load_json(dataset_path)

    before = len(data)
    removed = 0

    cleaned: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            cleaned.append(obj)
            continue

        t = obj.get("text")
        if not isinstance(t, str):
            cleaned.append(obj)
            continue

        key = t.strip() if use_strip_match else t

        # For strip-match mode, we must also strip the delete set once:
        # (simple approach: check both exact and stripped)
        if use_strip_match:
            if (t in texts_to_delete) or (t.strip() in {x.strip() for x in texts_to_delete}):
                removed += 1
                continue
        else:
            if t in texts_to_delete:
                removed += 1
                continue

        cleaned.append(obj)

    after = len(cleaned)
    save_json(out_path, cleaned)

    return {
        "dataset_path": dataset_path,
        "dups_report_path": dups_report_path,
        "out_path": out_path,
        "before": before,
        "after": after,
        "removed": removed,
        "expected_removed": len(texts_to_delete),
        "note": "If removed != expected_removed, some duplicate texts were not found or appeared multiple times."
    }


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_duplicate_ids_in_file(
    path: str,
    *,
    id_field: str = "id",
    text_field: str = "text",
    keep_examples_per_id: int = 20
) -> Dict[str, Any]:
    """
    Finds ids that appear more than once in the SAME file, along with their texts + indices.
    Returns a dict with summary + list of duplicates.
    """
    data = load_json(path)

    id_to_rows = defaultdict(list)
    skipped_no_id = 0
    skipped_bad_obj = 0

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            skipped_bad_obj += 1
            continue

        _id = obj.get(id_field)
        if _id is None or str(_id).strip() == "":
            skipped_no_id += 1
            continue

        _id = str(_id).strip()
        t = obj.get(text_field)
        if not isinstance(t, str):
            t = "" if t is None else str(t)

        id_to_rows[_id].append({
            "idx": idx,
            "text": t,
            "prompt_bn": obj.get("prompt_bn"),
        })

    duplicates = []
    for _id, rows in id_to_rows.items():
        if len(rows) > 1:
            # unique texts under this id (helps spot identical rows vs collisions)
            uniq_texts = sorted(set(r["text"] for r in rows if isinstance(r["text"], str)))

            duplicates.append({
                "id": _id,
                "count": len(rows),
                "unique_text_count": len(uniq_texts),
                "unique_texts": uniq_texts[:keep_examples_per_id],
                "rows": rows[:keep_examples_per_id],
            })

    duplicates.sort(key=lambda d: d["count"], reverse=True)

    return {
        "file": path,
        "total_items": len(data),
        "unique_ids": len(id_to_rows),
        "duplicate_id_count": len(duplicates),
        "skipped_no_id": skipped_no_id,
        "skipped_bad_obj": skipped_bad_obj,
        "duplicates": duplicates,
    }


def check_if_id_matches_with_sha1_of_text(path: str, normalize_whitespace: bool = True):
    """
    Returns 
    Total data length
    Toatl match count
    Total mismatch count
    """
    print("check_if_id_matches_with_sha1_of_text------->path: ", path)
    data = load_json(path)
    ids: Set[str] = set()
    matches_count = 0
    mismatches_count = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        t = obj.get("text")
        if isinstance(t, str):

            # if normalize_whitespace:
            #     txt_norm = " ".join(t.split())
            # else:
            #     txt_norm = t

            if  sha1_text(_norm_text(t)) == obj.get("id"):
                matches_count += 1
            else:
                mismatches_count += 1
        
            
    return len(data), matches_count, mismatches_count


def sha1_text_normalized(s: str, *, strip: bool = True, nfc: bool = True) -> str:
    """
    Use the SAME normalization here that you used when you originally created `id`.
    Most common: strip + NFC normalization.
    """
    if strip:
        s = s.strip()
    if nfc:
        s = unicodedata.normalize("NFC", s)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def collect_direct_ids_from_file(path: str) -> Set[str]:
    print("collect_sha1_ids_from_file -----> path:", path)
    data = load_json(path)
    print("data.len:", len(data))

    stored_ids: Set[str] = set()

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue

        stored = obj.get("id")
        stored_ids.add(stored)
    return stored_ids 

def collect_sha1_ids_from_file(path: str, sample_limit: int = 10, normalize_whitespace: bool =True) -> Set[str]:
    print("collect_sha1_ids_from_file -----> path:", path)
    data = load_json(path)
    print("data.len:", len(data))

    ids: Set[str] = set()

    valid_text = 0
    missing_id = 0
    matches = 0
    mismatches = 0
    mismatch_samples: List[Dict[str, Any]] = []

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue

        t = obj.get("text")
        if not isinstance(t, str) or not t.strip():
            continue

        valid_text += 1
        # computed = sha1_text_normalized(t)  # <--- normalized hash
        # if normalize_whitespace:
        #     txt_norm = " ".join(t.split())
        # else:
        #     txt_norm = t
        computed = sha1_text(_norm_text(t))
        
        ids.add(computed)

        stored = obj.get("id")
        if stored is None or str(stored).strip() == "":
            missing_id += 1
            continue

        stored_s = str(stored).strip()
        if stored_s == computed:
            matches += 1
        else:
            mismatches += 1
            if len(mismatch_samples) < sample_limit:
                mismatch_samples.append({
                    "idx": i,
                    "stored_id": stored_s,
                    "computed_sha1": computed,
                    "text_preview": t[:200],
                    "text_repr": repr(t[:200]),  # helps spot whitespace
                })

    print("valid_text:", valid_text)
    print("unique_ids(set size):", len(ids))
    print("missing_id:", missing_id)
    print("matches:", matches)
    print("mismatches:", mismatches)

    if mismatch_samples:
        print("\n--- mismatch samples (first few) ---")
        for s in mismatch_samples:
            print(s)

    return ids

def collect_stored_ids_from_file(path: str) -> Set[str]:
    """
    Returns a set of stored 'id' values as strings (if present).
    Useful to check if stored ids match sha1(text).
    """
    data = load_json(path)
    ids: Set[str] = set()
    for obj in data:
        if not isinstance(obj, dict):
            continue
        _id = obj.get("id")
        if _id is not None and str(_id).strip():
            ids.add(str(_id).strip())
    return ids


def pairwise_common_uncommon_ids(paths: List[str]) -> Dict[str, Any]:
    """
    For every pair of files, compute:
      - common_ids: IDs in both files
      - uncommon_ids: symmetric difference (in A or B but not both)
      - only_in_a / only_in_b
    Returns a dictionary keyed by "fileA||fileB".
    """
    per_file_ids: Dict[str, Set[str]] = {p: collect_direct_ids_from_file(p) for p in paths}

    results: Dict[str, Any] = {}
    for a, b in combinations(paths, 2):
        A = per_file_ids[a]
        B = per_file_ids[b]

        common = A & B
        uncommon = A ^ B          # symmetric difference
        only_a = A - B
        only_b = B - A

        key = f"{a}||{b}"
        results[key] = {
            "file_a": a,
            "file_b": b,
            "count_a": len(A),
            "count_b": len(B),
            "common_count": len(common),
            "uncommon_count": len(uncommon),
            "only_in_a_count": len(only_a),
            "only_in_b_count": len(only_b),
            "common_ids": sorted(common),
            "uncommon_ids": sorted(uncommon),
            "only_in_a": sorted(only_a),
            "only_in_b": sorted(only_b),
        }

    # Optional: also return per-file counts
    return {
        "num_files": len(paths),
        "per_file_counts": {p: len(s) for p, s in per_file_ids.items()},
        "pairs": results
    }

def write_common_uncommon_ids(common_ids: Set[str], uncommon_ids: Set[str],
                              out_common: str, out_uncommon: str) -> None:
    save_json(out_common, sorted(common_ids))
    save_json(out_uncommon, sorted(uncommon_ids))


def compare_common_uncommon_sha1(paths: List[str]) -> Dict[str, Any]:
    """
    Computes:
      - common_ids: sha1(text) IDs present in all files
      - uncommon_ids: sha1(text) IDs not present in all files
      - per_file_unique: ids that only appear in that one file (vs the union)
      - counts_across_files: how many files each id appears in
    """
    # per_file_ids: Dict[str, Set[str]] = {p: collect_sha1_ids_from_file(p) for p in paths}
    per_file_ids: Dict[str, Set[str]] = {p: collect_direct_ids_from_file(p) for p in paths}
    print("per_file_ids.items().len at beginning: ", len(per_file_ids.items()))
    for p, s in per_file_ids.items():
        print("p: ", p)
        print("s.len: ", len(s))

    summary = pairwise_common_uncommon_ids(paths)
    print("pairwise summary: ", summary)
    save_json("/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/out_pairwise_common_uncommon.json", summary)

    # Union and intersection
    print("Union and intersection")
    all_sets = list(per_file_ids.values())
    union_ids: Set[str] = set().union(*all_sets) if all_sets else set()
    common_ids: Set[str] = set.intersection(*all_sets) if all_sets else set()

    print("all_sets.len: ", len(all_sets))
    print("union_ids.len: ", len(union_ids))
    print("common_ids.len: ", len(common_ids))

    # Count in how many files each id appears
    counts = Counter()
    for p, s in per_file_ids.items():
        for _id in s:
            counts[_id] += 1

    uncommon_ids = union_ids - common_ids

    per_file_unique: Dict[str, List[str]] = {}
    for p, s in per_file_ids.items():
        others_union = union_ids - s
        # unique-to-this-file = in s but not in any other file
        unique_only = [i for i in s if counts[i] == 1]
        per_file_unique[p] = sorted(unique_only)

    
    print("per_file_ids.items().len at the end: ", len(per_file_ids.items()))
    for p, s in per_file_ids.items():
        print("p: ", p)
        print("s.len: ", len(s))

    write_common_uncommon_ids(common_ids, uncommon_ids,
                          "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/out_common_ids.json", 
                          "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/out_uncommon_ids.json")

    return {
        "num_files": len(paths),
        "per_file_counts": {p: len(s) for p, s in per_file_ids.items()},
        "union_count": len(union_ids),
        "common_count": len(common_ids),
        "uncommon_count": len(uncommon_ids),
        "common_ids": sorted(common_ids),
        "uncommon_ids": sorted(uncommon_ids),
        "counts_across_files": {k: int(v) for k, v in counts.items()},
        "per_file_unique_ids": per_file_unique,
    }

def sanity_check_stored_id_matches_sha1_text(path: str, limit: Optional[int] = 50) -> Dict[str, Any]:
    """
    Checks whether obj['id'] == sha1(obj['text']) for items in a file.
    Returns mismatch samples for debugging.
    """
    data = load_json(path)
    total = 0
    match = 0
    mismatches = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        t = obj.get("text")
        _id = obj.get("id")
        if not (isinstance(t, str) and t.strip()):
            continue
        if _id is None:
            continue
        total += 1
        computed = sha1_text(_norm_text(t))
        if str(_id).strip() == computed:
            match += 1
        else:
            if len(mismatches) < (limit or 50):
                mismatches.append({
                    "idx": i,
                    "stored_id": str(_id).strip(),
                    "computed_sha1": computed,
                    "text": t[:200]
                })

    return {
        "file": path,
        "checked": total,
        "matched": match,
        "mismatched": total - match,
        "mismatch_samples": mismatches
    }


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def find_ids_for_text(path: str, target_text: str) -> List[str]:
    data = load_json(path)
    ids = []
    for obj in data:
        if isinstance(obj, dict) and obj.get("text") == target_text:
            if "id" in obj and obj["id"] is not None:
                ids.append(str(obj["id"]))
    return ids

def check_id_consistency(target_text: str, paths: List[str]) -> None:
    expected = sha1_text(_norm_text(target_text))

    print(f"TEXT: {target_text}")
    print(f"sha1(text) expected: {expected}\n")

    per_file = {}
    for p in paths:
        ids = find_ids_for_text(p, target_text)
        per_file[p] = ids

    # Report
    all_ids_flat = []
    for p, ids in per_file.items():
        if not ids:
            print(f"[MISSING] {p}: no object found with this text")
        elif len(ids) == 1:
            print(f"[OK]      {p}: id={ids[0]}")
            all_ids_flat.append(ids[0])
        else:
            print(f"[DUPLICATE] {p}: multiple objects found; ids={ids}")
            all_ids_flat.extend(ids)

    # Consistency checks
    uniq = sorted(set(all_ids_flat))
    print("\n--- Consistency ---")
    if not all_ids_flat:
        print("No ids found in any file for this text.")
        return

    if len(uniq) == 1:
        print(f"All found ids match: {uniq[0]}")
    else:
        print(f"IDs differ across files: {uniq}")

    if expected in uniq and len(uniq) == 1:
        print("Also matches sha1(text). ✅")
    elif expected in uniq:
        print("At least one matches sha1(text), but not all match. ⚠️")
    else:
        print("None match sha1(text). ⚠️ (You likely hashed a normalized/other field.)")



def add_text_sha1_ids_in_file(
    input_path: str,
    out_path: str,
    text_field: str = "text",
    sha1_field: str = "id",
    normalize_whitespace: bool = True,
) -> Dict[str, int]:
    """
    Load a JSON list from `input_path`, compute SHA1 of each object's `text_field`,
    store it into `sha1_field`, and write back.

    - If out_path is None: overwrites input_path
    - Returns stats: {"total": N, "updated": K, "skipped_missing_text": M}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root to be a list, got {type(data)} in {input_path}")

    updated = 0
    skipped = 0

    for obj in data:
        if not isinstance(obj, dict):
            print("if not isinstance(obj, dict):")
            continue

        txt = obj.get(text_field)
        if not isinstance(txt, str) or not txt.strip():
            skipped += 1
            print("skipped += 1")
            continue

        # if normalize_whitespace:
        #     txt_norm = " ".join(txt.split())
        # else:
        #     txt_norm = txt

        obj[sha1_field] = sha1_text(_norm_text(txt))
        updated += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "total": len(data),
        "updated": updated,
        "skipped_missing_text": skipped,
    }


def convert_txt_to_json_array(input_path, output_path):
    json_array = []
    
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Clean up whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse each string as a JSON object
                data = json.loads(line)
                json_array.append(data)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")

    # Write the full list to a new file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(json_array, outfile, ensure_ascii=False, indent=4)
    
    print(f"Successfully processed {len(json_array)} items into {output_path}.")



def dedup(file_path):

    # Load the JSON array from your file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seen_texts = set()
    unique_data = []
    duplicate_data = []

    for idx, obj in enumerate(data):
        # print("obj: ", obj)
        

        try:
            if obj["text"] not in seen_texts:
                unique_data.append(obj)
                # break
                seen_texts.add(obj["text"])
            else:
                duplicate_data.append(obj)
        except Exception as e:
            print("idx: ", idx)
            print("obj: ", obj)
            print(e)

    print(f"Original objects: {len(data)}")
    print(f"Unique objects: {len(unique_data)}")
    print(f"Duplicate objects: {len(duplicate_data)}")

    return data, unique_data, duplicate_data


def format_toxic_gpt4o_data_into_json():
    # Read the file containing your line-separated JSON objects
    with open('/home/malam10/projects/ai-safety-bangla/final/data/gt_toxic_gpt4o_need_formating.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse each line as a JSON object
    json_objects = [json.loads(line.strip()) for line in lines if line.strip()]

    # Convert the list of objects into a single JSON array string
    json_array = json.dumps(json_objects, ensure_ascii=False, indent=2)

    # Save to a new JSON file
    with open('/home/malam10/projects/ai-safety-bangla/final/data/gt_toxic_gpt4o_formatted.json', 'w', encoding='utf-8') as f:
        f.write(json_array)

    print("Conversion complete!")


def copy_json_file(src_path, dst_path):
    """
    Copy JSON content from src_path to dst_path.
    Creates dst_path if it does not exist.
    """

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # ---- Read source JSON ----
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- Write to destination JSON ----
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Copied content from '{src_path}' → '{dst_path}'")


def merge_json_list(L1: list, L2: list, output_path: str):
    
    # Combine the lists using '+'
    combined_data = L1 + L2
    
    # Save the merged list
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # ensure_ascii=False is CRITICAL for Bangla text
        json.dump(combined_data, outfile, ensure_ascii=False, indent=4)
        
    print(f"Done! Merged {len(L1)} and {len(L2)} items. Total: {len(combined_data)}")

def merge_json_files(file1_path, file2_path, output_path):
    # Load the first list
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    # Load the second list
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # Combine the lists using '+'
    combined_data = data1 + data2
    
    # Save the merged list
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # ensure_ascii=False is CRITICAL for Bangla text
        json.dump(combined_data, outfile, ensure_ascii=False, indent=4)
        
    print(f"Done! Merged {len(data1)} and {len(data2)} items. Total: {len(combined_data)}")



def combine_toxic_gemini_collected_and_formatted(file1_path, file2_path, output_path):
    # Run it
    # merge_json_files('part1.json', 'part2.json', 'merged_dataset.json')
    merge_json_files(file1_path=file1_path, file2_path=file2_path, output_path=output_path)



def load_json_any(path: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # JSONL
    if "\n" in text and text.lstrip().startswith("{"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # JSON array
    return json.loads(text)


def normalize_object(obj: dict, model_name: str) -> dict:
    """
    1. Adds model_name at root level
    2. Moves judge fields into closed_llm: [ {...} ]
    3. Removes judge fields from root
    """

    # ---- add model name at root ----
    obj["model_name"] = model_name

    # ---- collect judge fields ----
    judge_entry = {}
    for field in FIELDS_TO_MOVE:
        if field in obj:
            judge_entry[field] = obj[field]

    # ---- move into closed_llm if any exist ----
    if judge_entry:
        obj["closed_llm"] = [judge_entry]

        # remove from root
        for field in FIELDS_TO_MOVE:
            obj.pop(field, None)

    return obj


def transform_dataset_move_fields_into_closeed_llm_field_by_file(
    input_path: str,
    output_path: str,
    model_name: str
):
    data = load_json_any(input_path)

    transformed = []
    for obj in data:
        transformed.append(normalize_object(obj, model_name))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(transformed)} objects to: {output_path}")


def update_closed_llm_model_name(filepath, new_model_name):
    """
    Update the `model_name` field inside each object's `closed_llm` list.

    Parameters
    ----------
    filepath : str
        Path to JSON file
    new_model_name : str
        New model name to set (e.g., 'gpt-4o')
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # ---- Read JSON ----
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    def update_object(obj):
        if not isinstance(obj, dict):
            return obj

        closed_llm = obj.get("closed_llm")

        if isinstance(closed_llm, list):
            for entry in closed_llm:
                if isinstance(entry, dict) and "model_name" in entry:
                    entry["model_name"] = new_model_name

        return obj

    # ---- Apply update ----
    if isinstance(data, list):
        data = [update_object(item) for item in data]
    elif isinstance(data, dict):
        data = update_object(data)
    else:
        raise ValueError("JSON root must be a dict or list of dicts")

    # ---- Write back ----
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated model_name → '{new_model_name}' in {filepath}")


def update_field_value(filepath, field_name, new_value):
    """
    Update the value of a field in every JSON object stored in the file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file
    field_name : str
        Field whose value should be updated
    new_value : any
        New value to assign to the field
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # ---- Read JSON ----
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    def update_in_object(obj):
        if isinstance(obj, dict) and field_name in obj:
            obj[field_name] = new_value
        return obj

    # ---- Apply update ----
    if isinstance(data, list):
        data = [update_in_object(item) for item in data]
    elif isinstance(data, dict):
        data = update_in_object(data)
    else:
        raise ValueError("JSON root must be a dict or list of dicts")

    # ---- Write back ----
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated value of '{field_name}' in {filepath}")

def rename_field_name(filepath, old_field, new_field):
    """
    Rename a field in every JSON object stored in the file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file
    old_field : str
        Existing field name to rename
    new_field : str
        New field name
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # ---- Read JSON ----
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- Helper to rename field in a single object ----
    def rename_in_object(obj):
        if isinstance(obj, dict) and old_field in obj:
            obj[new_field] = obj.pop(old_field)
        return obj

    # ---- Apply rename ----
    if isinstance(data, list):
        data = [rename_in_object(item) for item in data]
    elif isinstance(data, dict):
        data = rename_in_object(data)
    else:
        raise ValueError("JSON root must be a dict or list of dicts")

    # ---- Write back to file ----
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Renamed '{old_field}' → '{new_field}' in {filepath}")



def transform_dataset_move_fields_into_closeed_llm_field_by_list(
    data: list,
    model_name: str
):

    transformed = []
    for obj in data:
        transformed.append(normalize_object(obj, model_name))
    return transformed


def extract_max_harm_variant(obj: dict) -> dict:
    """
    From obj["variants"], pick the variant with highest harm_score
    and promote it to root-level prompt_bn + harm_score.
    """

    variants = obj.get("variants", [])

    if isinstance(variants, list) and variants:
        # pick variant with max harm_score
        max_variant = max(
            variants,
            key=lambda v: float(v.get("harm_score", 0))
        )

        obj["prompt_bn"] = max_variant.get("Prompt")
        obj["harm_score"] = max_variant.get("harm_score")

    # remove unwanted fields
    obj.pop("prompt", None)
    obj.pop("variants", None)

    return obj

def find_items_with_variant_fields(input_path):
    """
    Returns list of objects that HAVE the 'variants' field
    (and it is a non-empty list).
    """
    data = load_json_any(input_path)

    L = [
        obj for obj in data
        if isinstance(obj, dict)
        and "variants" in obj
        and isinstance(obj["variants"], list)
        and len(obj["variants"]) > 0
    ]

    return L


def find_items_without_variant_fields(input_path):
    """
    Returns list of objects that DO NOT HAVE the 'variants' field
    or where 'variants' is empty / invalid.
    """
    data = load_json_any(input_path)

    L = [
        obj for obj in data
        if not (
            isinstance(obj, dict)
            and "variants" in obj
            and isinstance(obj["variants"], list)
            and len(obj["variants"]) > 0
        )
    ]

    return L

def transform_find_variant_with_highest_harm_score(input_path: str, output_path: str):
    data = load_json_any(input_path)

    transformed = []
    for obj in data:
        transformed.append(extract_max_harm_variant(obj))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"Transformed {len(transformed)} objects")
    print(f"Saved to: {output_path}")


def transform_find_variant_with_highest_harm_score_from_list(data: list):
    """
    Takes a list of objects.
    For each object:
      - finds the variant with highest harm_score
      - promotes it to root-level prompt_bn + harm_score
      - removes 'prompt' and 'variants'
    Returns a new transformed list.
    """

    transformed = []
    for obj in data:
        transformed.append(extract_max_harm_variant(obj))

    return transformed
