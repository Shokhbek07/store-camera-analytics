# ONNX Runtime comparison (E15)

- `eval/`: `scripts/evaluate.py --rerun` with `model.weights: models/yolo26s_384x640_b2.onnx` on the Mac GPU (CoreML, MLProgram); identical to the PyTorch `results/eval/` of the same configuration (commit ef68f83, before the queue changes of E16) apart from wall-clock timestamps.
- `detector_mac_2026-11-11.jsonl`, `detector_docker_cpu_2026-11-11.json`, `ort_cpu_threads_and_coreml_units_2026-11-11.txt`: the detector-only timings in E15, from a first version of `scripts/compare_detectors.py` on a machine that was not quiet (load as stated in E15).
- `benchmark_mac_back_to_back_2026-11-11.json`: `scripts/benchmark.py` once per runtime, one after the other; the load rose from 3.7 to 38.6 during the second, so the two are not comparable (E15).
- `benchmark_mac_alternating_2026-11-11.json`: `scripts/benchmark.py --compare` with both runtimes in turn, 3 repeats, the load recorded per run, medians at the top (E15).
- `benchmark_docker_alternating_under_load_2026-11-11.json`: `scripts/benchmark_docker.sh 120 3` in the container, both runtimes in turn, while other projects' containers used about 7.5 of the VM's 10 CPUs (E15).
