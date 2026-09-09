#!/usr/bin/env python3
"""Final stats for SHTM attribute track: agree rate + failures from attr_track.jsonl."""
import collections
import json

OUT = "/tmp/shtm_s1/attr_track.jsonl"

ok = fail = 0
agree_all = collections.defaultdict(lambda: [0, 0])  # (cls, key) -> [match, disagree]
obj_agree = [0, 0]  # [all-common-keys-agree objects, judged objects]
conf_low = collections.Counter()
new_vals = collections.defaultdict(collections.Counter)
illegal_new = collections.Counter()

for l in open(OUT):
    d = json.loads(l)
    if "error" in d:
        fail += 1
        continue
    ok += 1
    per_key_match = []
    for k, v in d["new"].items():
        new_vals[k][v["value"]] += 1
        if v["conf"] is not None and v["conf"] < 0.9:
            conf_low[k] += 1
        if k in d["old"]:
            m = v["value"] == d["old"][k]["value"]
            agree_all[(d["cls"], k)][0 if m else 1] += 1
            per_key_match.append(m)
        if k == "illegal":
            illegal_new[v["value"]] += 1
    if per_key_match:
        obj_agree[0 if all(per_key_match) else 1] += 1

print(f"objects ok={ok} fail={fail} total={ok + fail}")
print(f"object-level agree: {obj_agree[0]}/{obj_agree[0] + obj_agree[1]} = "
      f"{obj_agree[0] / max(1, obj_agree[0] + obj_agree[1]):.1%}")
print("\nper-key agree (new vs old):")
for (cls, k), (match, dis) in sorted(agree_all.items()):
    tot = match + dis
    print(f"  cls{cls} {k:8s} {match}/{tot} = {match / tot:.1%}" + (f"  (disagree {dis})" if dis else ""))
print("\nnew value distribution:")
for k, c in sorted(new_vals.items()):
    print(f"  {k:8s} {dict(sorted(c.items()))}")
print("\nconf<0.9 counts:", dict(conf_low) or "{}")
if illegal_new:
    print(f"illegal new: {dict(illegal_new)}  (yes rate={illegal_new.get(1, 0) / sum(illegal_new.values()):.1%})")
