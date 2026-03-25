# study/ — Zero-to-Master Learning Guide

Self-contained AI learning guide. 25 chapters + 5 appendices, ~43K words. No external materials needed.

## Structure

```
Part 1: Foundations (01-03)     — AI, neural nets, transformers
Part 2: Building Blocks (04-07) — tokens/embeddings/RoPE, audio, images, normalization
Part 3: Advanced (08-11)        — decoder LLMs, GQA/Flash, MoE, vector quantization
Part 4: μOmni Architecture (12-17) — system overview, each component in detail
Part 5: Training (18-21)        — data prep, 5-stage pipeline, optimization, debugging
Part 6: Deployment (22-25)      — setup, inference, export, testing
Appendices A-E                  — math, papers, config reference, code structure, customization
```

## Writing Rules (for updating/adding chapters)
- Every concept gets a real-life analogy (kitchen, library, orchestra, hospital, etc.)
- ASCII diagrams for all data flows — include actual tensor shapes from μOmni
- Use μOmni's real config values (d=384, layers=8, heads=6, etc.)
- No external links — everything explained in-place
- No duplicate content — each concept explained in exactly ONE chapter
- Reference other chapters with "As we saw in Chapter X..." not by re-explaining
- Keep paragraphs short (3-5 lines max)
- Commands must be copy-paste ready for Windows bash

## Topic Ownership (to prevent duplication)
- Attention mechanism: Chapter 03 ONLY
- RoPE: Chapter 04 ONLY
- Flash Attention + GQA: Chapter 09 ONLY
- MoE: Chapter 10 ONLY
- RVQ + vocoders: Chapter 11 ONLY
- Each μOmni component: its own chapter in Part 4 (12-17)
- All training stages: Chapter 19 ONLY (not split across files)
- All math formulas: Appendix A ONLY
- All config parameters: Appendix C ONLY

## Pending Additions
- **HuggingFace Integration chapter needed**: Cover `from_pretrained` workflow, `MuOmniForCausalLM` / `MuOmniMultimodalModel` classes, export format (flat keys vs prefixed keys), and testing with `test_hf_text.py` / `test_hf_multimodal.py`
