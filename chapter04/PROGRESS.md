# Chapter 4 progress

Status: complete (builds to 24 pages, no undefined references or citations).

- [x] Brief, ch1 template, slide text, transcript (2025-10-03 "Interpreting neural networks" to end) read
- [x] Slides viewed: 2026 deck (all 38), 2023 deck (IG slides 30-36 used; rest matches 2026), deepLIFT exercise sheet (2 examples)
  - The stock renderer omits node-link diagrams (shapes/connectors); a scratch copy was extended to draw them.
- [x] Sections: why interpret; perturbation/ISM; gradients (toy CNN); deconvnet/guided backprop (3x3 rules, positive paths, limitations, saturation/thresholds); DeepLIFT (multipliers, chain rule, rules, worked example); integrated gradients (+expected gradients); Shapley/SHAP; simulated-sequence example; summary; 7 exercises
- [x] Computed figures (scripts/make_figures.py, scripts/attribution.py): secant, ig_saturation, sequence_attributions; check_toy_examples() asserts every worked-example and exercise answer
- [x] TikZ: power vs interpretability, attribution schematic, saturation net, toy CNN (4 panels), 3x3 ReLU rules, positive paths, threshold net, DeepLIFT saturation
- [x] Slide figure: figures/slides/occlusion_zeiler_fergus.png (permission needed)
- [x] README with provenance table
