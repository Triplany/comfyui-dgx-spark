# Workflow fixes

Just a fixer for some flux1 workflows I use. Included as it may help others — not Spark-related.

## fix-flux-lora-clip.py

Fixes Flux LoRA workflows that silently drop kohya-trained text-encoder weights.

The `LoraLoaderModelOnly` node (common in Flux templates and stacking patterns) only patches the UNet — never the CLIP. If your LoRA was trained with text encoder training enabled (kohya `--network_train_text_encoder`, producing `lora_te1_*` keys), those weights never get applied and you'll see hundreds of `lora key not loaded: lora_te1_text_model_encoder_layers_*` warnings on every gen. This script converts those nodes to `LoraLoader` and threads CLIP through the chain so the text-encoder weights actually load.

## Usage

```bash
# Dry-run to see what would change
python workflows/fix-flux-lora-clip.py --dry-run /path/to/workflow.json

# Apply (creates a timestamped .bak file beside the original)
python workflows/fix-flux-lora-clip.py /path/to/workflow.json

# Multiple workflows at once
python workflows/fix-flux-lora-clip.py /path/to/workflow1.json /path/to/workflow2.json
```

Idempotent — workflows with no `LoraLoaderModelOnly` nodes are skipped, no changes made.

After applying, reload the workflow in ComfyUI's web UI (Workflow → Open) and the `lora_te1_*` warnings should be gone on the next gen.

## Limitations

- Only handles single-`DualCLIPLoader`, single-`CLIPTextEncodeFlux` topologies (the standard Flux pattern). Multi-fan-out CLIP graphs need manual wiring after the script runs (the script will warn).
- Backups are `*.bak.YYYYMMDD_HHMMSS` next to the original — delete after you've confirmed the patch works.
