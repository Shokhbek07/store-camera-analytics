# ONNX Runtime comparison (E15)

- `eval/`: `scripts/evaluate.py --rerun` with `model.weights: models/yolo26s_384x640_b2.onnx` on the Mac GPU (CoreML, MLProgram); identical to the PyTorch `results/eval/` of the same configuration (commit ef68f83, before the queue changes of E16) apart from wall-clock timestamps.
- `detector_mac_2026-11-11.jsonl`, `detector_docker_cpu_2026-11-11.json`, `ort_cpu_threads_and_coreml_units_2026-11-11.txt`: the detector-only timings in E15, from a first version of `scripts/compare_detectors.py` on a machine that was not quiet (load as stated in E15).
