# Durable project rules

- Preserve the original Progress 1–2 PDF, EDA notebook, source, results and figures. Never change raw data, labels or historical evidence without explicit instruction.
- Maintain the academic comparison: three-block CNN baseline and MobileNetV3-Small transfer learning; no replacement architecture or web application in Progress 3.
- Never fabricate metrics or execution evidence. Do not train or perform final evaluation without explicit authorization.
- Prevent train/test leakage: review near-duplicate candidates, use stratified group-aware splitting, fit data-derived statistics on training only, augment training only, reserve testing for final evaluation.
- Keep model implementations reusable in src/models and import them in notebooks.
- Record actual dependency versions, commands, test results and limitations. Ordinary tests use CPU synthetic inputs and never download weights or data.
- Augmentation and class weighting are separate controlled experiments, not unconditional defaults.
- Do not commit or push unless explicitly requested.
