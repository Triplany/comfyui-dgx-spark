#!/usr/bin/env python3
"""
Fix Flux LoRA workflows: convert LoraLoaderModelOnly to LoraLoader and
thread CLIP through the LoRA chain so kohya `lora_te1_*` (CLIP-L) keys
get applied.

Idempotent: skips workflows that have no LoraLoaderModelOnly nodes.
"""
import json
import os
import sys
import shutil
import argparse
from datetime import datetime


def find_clip_source_link(wf_links, dual_clip_node_id, clip_consumer_node_id):
    """Return the existing link_id that goes from DualCLIPLoader -> CLIP consumer (the link we'll re-route)."""
    for L in wf_links:
        if len(L) >= 6 and L[1] == dual_clip_node_id and L[3] == clip_consumer_node_id and L[5] == "CLIP":
            return L
    return None


def order_lora_chain(nodes, lora_node_ids, links_index):
    """
    Walk the LoraLoaderModelOnly chain in MODEL flow order.
    Returns ordered list of node IDs (first → last).
    """
    # build map: node_id -> (model_input_link, [model_output_link, ...])
    node_map = {n["id"]: n for n in nodes}
    # Find the first lora: its model input link source is NOT another lora node
    lora_set = set(lora_node_ids)
    first_id = None
    for nid in lora_node_ids:
        n = node_map[nid]
        # find model input link
        model_link_id = None
        for inp in n.get("inputs", []):
            if inp.get("name") == "model":
                model_link_id = inp.get("link")
                break
        if model_link_id is None:
            continue
        # find link source
        L = links_index.get(model_link_id)
        if L is None:
            continue
        src_node_id = L[1]
        if src_node_id not in lora_set:
            first_id = nid
            break
    if first_id is None:
        # fallback: assume the first one in the list
        first_id = lora_node_ids[0]

    # Walk forward following MODEL output links to next lora
    ordered = [first_id]
    cur = first_id
    while True:
        n = node_map[cur]
        next_id = None
        for out in n.get("outputs", []):
            if out.get("name") == "MODEL":
                for lid in out.get("links") or []:
                    L = links_index.get(lid)
                    if L is None:
                        continue
                    dst = L[3]
                    if dst in lora_set and dst not in ordered:
                        next_id = dst
                        break
            if next_id:
                break
        if next_id is None:
            break
        ordered.append(next_id)
        cur = next_id
    return ordered


def fix_workflow(path, dry_run=False, verbose=True):
    with open(path) as f:
        wf = json.load(f)

    nodes = wf.get("nodes", [])
    links = wf.get("links", [])
    last_link = wf.get("last_link_id", 0)

    lora_nodes = [n for n in nodes if n.get("type") == "LoraLoaderModelOnly"]
    if not lora_nodes:
        if verbose:
            print(f"  [no-op] {path}: no LoraLoaderModelOnly nodes")
        return False

    dual_clip_nodes = [n for n in nodes if n.get("type") == "DualCLIPLoader"]
    clip_consumers = [n for n in nodes if n.get("type") in ("CLIPTextEncodeFlux", "CLIPTextEncode")]

    if not dual_clip_nodes:
        print(f"  [skip] {path}: no DualCLIPLoader found", file=sys.stderr)
        return False
    if not clip_consumers:
        print(f"  [skip] {path}: no CLIP consumer found", file=sys.stderr)
        return False
    if len(dual_clip_nodes) > 1:
        print(f"  [skip] {path}: multiple DualCLIPLoader nodes ({len(dual_clip_nodes)}) — manual fix needed", file=sys.stderr)
        return False
    if len(clip_consumers) > 1:
        # We'll thread through and have the LAST output go to the first consumer; manual fan-out not supported
        print(f"  [warn] {path}: {len(clip_consumers)} CLIP consumers — only first will be threaded; check manually after", file=sys.stderr)

    dual = dual_clip_nodes[0]
    consumer = clip_consumers[0]

    links_index = {L[0]: L for L in links}
    # Find the existing CLIP link from dual -> consumer
    src_link = find_clip_source_link(links, dual["id"], consumer["id"])
    if src_link is None:
        print(f"  [skip] {path}: no direct DualCLIPLoader→CLIPTextEncode link found (already custom-wired?)", file=sys.stderr)
        return False

    lora_ids = [n["id"] for n in lora_nodes]
    chain = order_lora_chain(nodes, lora_ids, links_index)
    if verbose:
        print(f"  {os.path.basename(path)}:")
        print(f"    LoRA chain order: {chain}")
        print(f"    CLIP src link {src_link[0]}: node{src_link[1]}[{src_link[2]}] -> node{src_link[3]}[{src_link[4]}]")

    # Build new link plan:
    # - existing src_link (id 4 typically): src stays (dual,0), dst changes to (chain[0], 1)
    # - new link: chain[0].CLIP -> chain[1].clip
    # - new link: chain[1].CLIP -> chain[2].clip
    # - ...
    # - new link: chain[-1].CLIP -> consumer.clip(slot 0)

    # 1) Mutate src_link to point to chain[0] clip input (slot index 1 — after model)
    src_link[3] = chain[0]
    src_link[4] = 1

    # 2) Generate intermediate + final links
    new_links = []
    next_id = last_link + 1
    for i in range(len(chain)):
        if i + 1 < len(chain):
            nl = [next_id, chain[i], 1, chain[i + 1], 1, "CLIP"]
        else:
            # last: connect to consumer's clip input. Find consumer's CLIP input slot index
            slot = None
            for idx, inp in enumerate(consumer.get("inputs", [])):
                if inp.get("name") == "clip" and inp.get("type") == "CLIP":
                    slot = idx
                    break
            if slot is None:
                slot = 0
            nl = [next_id, chain[i], 1, consumer["id"], slot, "CLIP"]
            # update consumer node's input link
            consumer["inputs"][slot]["link"] = next_id
        new_links.append(nl)
        next_id += 1

    # 3) Mutate each LoRA node
    node_by_id = {n["id"]: n for n in nodes}
    chain_link_ids = [src_link[0]] + [nl[0] for nl in new_links]
    # chain_link_ids: [link_to_first_lora_clip, link_first_to_second, ..., link_last_to_consumer]
    for i, nid in enumerate(chain):
        n = node_by_id[nid]
        n["type"] = "LoraLoader"
        # properties update
        if "properties" in n:
            n["properties"]["Node name for S&R"] = "LoraLoader"
        # add clip input
        clip_in_link = chain_link_ids[i]
        n["inputs"] = list(n.get("inputs", []))
        # only add if not already present (idempotent)
        if not any(inp.get("name") == "clip" for inp in n["inputs"]):
            n["inputs"].append({"name": "clip", "type": "CLIP", "link": clip_in_link})
        else:
            # update link
            for inp in n["inputs"]:
                if inp.get("name") == "clip":
                    inp["link"] = clip_in_link
                    break
        # add CLIP output
        clip_out_link = chain_link_ids[i + 1]
        n["outputs"] = list(n.get("outputs", []))
        if not any(out.get("name") == "CLIP" for out in n["outputs"]):
            n["outputs"].append({"name": "CLIP", "type": "CLIP", "slot_index": 1, "links": [clip_out_link]})
        else:
            for out in n["outputs"]:
                if out.get("name") == "CLIP":
                    out["links"] = [clip_out_link]
                    break
        # widgets_values: append strength_clip = strength_model if only 2 entries
        wv = n.get("widgets_values") or []
        if len(wv) == 2:
            n["widgets_values"] = [wv[0], wv[1], wv[1]]  # strength_clip = strength_model
        elif len(wv) == 3:
            pass  # already correct shape
        else:
            print(f"    [warn] node {nid}: unexpected widgets_values shape: {wv}")

    # 4) Add new links to links[]
    for nl in new_links:
        links.append(nl)

    # 5) Bump last_link_id
    wf["last_link_id"] = next_id - 1

    if verbose:
        print(f"    Patched: {len(chain)} LoRA nodes, {len(new_links)} new links, last_link_id now {wf['last_link_id']}")

    if dry_run:
        return True

    # Backup
    bak = path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, bak)
    if verbose:
        print(f"    Backup: {bak}")

    # Write
    with open(path, "w") as f:
        json.dump(wf, f, indent=2)
    if verbose:
        print(f"    Wrote: {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    for p in args.paths:
        try:
            fix_workflow(p, dry_run=args.dry_run)
        except Exception as e:
            print(f"  [error] {p}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
