# Profiling vLLM-Omni
## profiling hooks for omni&vllm&diffusion pipeline

## 1.Usage of Log Statistics for Single-Pipeline Diffusion Scheduling  


In this project, **tasks such as text-to-image and text-to-video follow a single-pipeline diffusion scheduling paradigm**.  
Each request **triggers the diffusion pipeline as a whole**, executing text encoding, denoising iterations, and decoding in a tightly coupled, end-to-end manner.

- The entire workflow is launched **in one shot** via `Omni.generate(...)`.
- Execution proceeds sequentially within the diffusion engine.
- Performance and behavior can be directly inspected through:
  - **Diffusion-level logs** (e.g., denoising steps, post-processing),
  - **vLLM runtime logs** (e.g., worker startup, device allocation).
> **Text-to-Image / Text-to-Video** → *Single diffusion pipeline, single execution path*  
### The log usage method is as follows:
### 1.Print the vllm feature.
1)vllm feature integration
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```
2)Run script(Taking image_to_image as an example, the usage method for other models is the same.):

```python
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0
```
We can see the vLLM logs in the console and the diffusion logs in path/omni_diffusion_stats/omni_diffusion_%Y%m%d_%H%M%S_xx_pidxxxx.jsonl.

```json
DEBUG 12-17 09:21:42 [plugins/__init__.py:28] No plugins for group vllm.platform_plugins found.
DEBUG 12-17 09:21:42 [platforms/__init__.py:34] Checking if TPU platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 12-17 09:21:42 [platforms/__init__.py:58] Checking if CUDA platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:78] Confirmed CUDA platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:106] Checking if ROCm platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 12-17 09:21:42 [platforms/__init__.py:127] Checking if XPU platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 12-17 09:21:42 [platforms/__init__.py:153] Checking if CPU platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:58] Checking if CUDA platform is available.
DEBUG 12-17 09:21:42 [platforms/__init__.py:78] Confirmed CUDA platform is available.
INFO 12-17 09:21:42 [platforms/__init__.py:216] Automatically detected platform cuda.
DEBUG 12-17 09:21:47 [compilation/decorators.py:155] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.qwen3_moe.Qwen3MoeModel'>: ['input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds']
WARNING 12-17 09:21:47 [mooncake_connector.py:18] Mooncake not available, MooncakeOmniConnector will not work
DEBUG 12-17 09:21:47 [factory.py:35] Registered connector: MooncakeConnector
DEBUG 12-17 09:21:47 [factory.py:35] Registered connector: SharedMemoryConnector
DEBUG 12-17 09:21:48 [distributed/device_communicators/shm_broadcast.py:313] Connecting to ipc:///tmp/5c30e5fa-26de-43e1-bd35-d551269b0fe2
DEBUG 12-17 09:21:48 [distributed/device_communicators/shm_broadcast.py:243] Binding to ipc:///tmp/7c1c23a5-2d1c-4f83-a6f2-36d8c4c71644
INFO 12-17 09:21:48 [distributed/device_communicators/shm_broadcast.py:289] vLLM message queue communication handle: Handle(local_reader_ranks=[0], buffer_handle=(1, 10485760, 10, 
INFO 12-17 09:22:16 [diffusers_loader.py:214] Loading weights took 17.82 seconds
INFO 12-17 09:22:16 [gpu_worker.py:81] Model loading took 53.7462 GiB and 27.811149 seconds
INFO 12-17 09:22:16 [gpu_worker.py:86] Worker 0: Model loaded successfully.
INFO 12-17 09:22:16 [gpu_worker.py:237] Worker 0: Scheduler loop started.
INFO 12-17 09:22:16 [gpu_worker.py:175] Worker 0 ready to receive requests via shared memory
DEBUG 12-17 09:22:16 [diffusion_engine.py:147] All workers are ready
DEBUG 12-17 09:22:16 [distributed/device_communicators/shm_broadcast.py:313] Connecting to ipc:///tmp/7c1c23a5-2d1c-4f83-a6f2-36d8c4c71644
INFO 12-17 09:22:16 [scheduler.py:45] SyncScheduler initialized result MessageQueue
INFO 12-17 09:22:16 [omni_diffusion.py:114] OmniDiffusion initialized: model=path/models/Qwen-Image-Edit, class=QwenImageEditPipeline, init_ms=36702.19
Pipeline loaded
```
---
## omni_diffusion_%Y%m%d_%H%M%S_xx_pidxxxx.jsonl
```json
{"model": "path/models/Qwen-Image", "model_class": "QwenImagePipeline", "init_ms": 17562.917941002524, "event": "engine_load", "ts": 1766019712.405319, "pid": 18635, "host": "xxxx"}
{"n_requests": 1, "prompt_chars": 28, "height": 1024, "width": 1024, "generator": "<torch._C.Generator object at 0x7fc71d96e8f0>", "true_cfg_scale": 4.0, "num_inference_steps": 50, "num_outputs_per_prompt": 1, "event": "request_scheduled", "ts": 1766019712.405916, "pid": 18635, "host": "xxxx"}
{"n_requests": 1, "total_ms": 42437.41700099781, "diffusion_total_ms": 42437.13191100687, "denoise_avg_ms": 848.7426382201375, "input_tokens": 28, "input_tokens_per_s": 0.6597951048562086, "event": "request_finished", "ts": 1766019754.8433862, "pid": 18635, "host": "xxxx"}

```
### 2.The vllm feature is not printed..
Run script:
```python
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0
```
We can see the vLLM logs in the console .The characteristics of diffusion models will still be statistically analyzed.

```json
INFO 12-17 09:28:58 [__init__.py:216] Automatically detected platform cuda.
WARNING 12-17 09:29:03 [mooncake_connector.py:18] Mooncake not available, MooncakeOmniConnector will not work
Loaded input image from input.png (size: (514, 556))
INFO 12-17 09:29:06 [shm_broadcast.py:289] vLLM message queue communication handle: Handle(local_reader_ranks=[0], buffer_handle=(1, 10485760, 10, 'psm_0c8120b1'), local_subscribe_addr='ipc:///tmp/7f7c25ae-cf87-4c4d-b79d-17cbb4ea00e2', remote_subscribe_addr=None, remote_addr_ipv6=False)
INFO 12-17 09:29:06 [diffusion_engine.py:92] Starting server...
.......
INFO 12-17 09:29:26 [diffusion_engine.py:43] Pre-processing completed in 0.0564 seconds
INFO 12-17 09:30:26 [shm_broadcast.py:466] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation).
INFO 12-17 09:31:17 [diffusion_engine.py:48] Generation completed successfully.
INFO 12-17 09:31:17 [diffusion_engine.py:53] Post-processing completed in 0.0651 seconds
INFO 12-17 09:31:17 [omni_diffusion.py:177] request_finished: n_requests=1, total_ms=111360.70
INFO 12-17 09:31:17 [omni_diffusion.py:184] request_scheduled: n_requests=1, kwargs_keys=['pil_image', 'negative_prompt', 'generator', 'true_cfg_scale', 'num_inference_steps', 'num_outputs_per_prompt'], kwargs_detail={'generator': '<torch._C.Generator object at 0x7f0c71d328d0>', 'true_cfg_scale': 4.0, 'num_inference_steps': 50, 'num_outputs_per_prompt': 1}
INFO 12-17 09:31:17 [omni_diffusion.py:190] OMNI_DIFFUSION_METRICS {"prompt_chars": 103, "input_tokens": 103, "input_tokens_per_s": 0.9249223665998467, "num_inference_steps": 50, "diffusion_total_ms": 111360.43418200097, "denoise_avg_ms": 2227.2086836400194, "total_ms": 111360.69763200067}
Total generation time: 111.3614 seconds (111361.44 ms)
Saved edited image to path/vllm-omni/examples/offline_inference/image_to_image/output_image_edit.png
INFO 12-17 09:31:17 [gpu_worker.py:190] Worker 0: Received shutdown message
INFO 12-17 09:31:17 [gpu_worker.py:214] event loop terminated.
INFO 12-17 09:31:17 [gpu_worker.py:114] Worker 0: Destroyed process group
INFO 12-17 09:31:17 [gpu_worker.py:245] Worker 0: Shutdown complete.

```
---
## 2.Usage of Log Statistics for Multi-Process, Multi-Stage Scheduling  

In contrast, **Qwen2.5-Omni and Qwen3-Omni adopt a multi-process, multi-stage scheduling model driven by OmniLLM**.  
Rather than executing a single pipeline, the system decomposes the task into **multiple stages**, each running as an independent process.

- The core abstraction is a **stage-based pipeline**:
  - Each stage performs a specific function (e.g., reasoning, generation, modality transformation).
  - Stages are connected via **inter-process communication (IPC)**.
- Scheduling is pipeline-oriented:
  - Downstream stages are activated once upstream stages complete.
  - Multiple stages can overlap in time, enabling **pipeline parallelism**.
- System behavior can be observed through:
  - **Omni-level logs** (stage transitions and orchestration),
  - **Diffusion logs** (if diffusion is involved in a stage),
  - **vLLM logs** (process lifecycle, execution and resource usage).
> **Qwen2.5-Omni / Qwen3-Omni** → *Multi-process, multi-stage pipeline with explicit scheduling*

1. Setting the log switch.:

```python
    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,#Setting  enable_stats=True 
        log_file=(os.path.join(log_dir, "omni_llm_pipeline.log") if args.enable_stats else None)
    )
```
or
```python
    omni_llm = Omni(
        model=model_name,
        log_stats=True 
        log_file=os.path.join(log_dir, "omni_llm_pipeline.log") 
    )

```
2. Run  script:

```bash
sh run_multiple_prompts.sh
```
or
```bash
run_single_prompt.sh
```
4. vllm feature integration
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```
We can see the debug log（vllm+omni+diffusion）in omni_llm_pipeline.log:
```json
2025-12-16 01:24:23,021 [PID:17815] DEBUG: [Orchestrator] generate() called
2025-12-16 01:24:23,021 [PID:17815] DEBUG: [Orchestrator] Seeding 1 requests into stage-0
2025-12-16 01:24:23,022 [PID:17815] DEBUG: [Orchestrator] Enqueued request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-0
2025-12-16 01:24:23,023 [PID:17815] DEBUG: [Orchestrator] Entering scheduling loop: total_requests=1, stages=3
2025-12-16 01:24:26,527 [PID:17815] INFO: [StageMetrics] stage=0 req=0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num_tokens_out': 52, 'stage_gen_time_ms': 3490.6439781188965, 'batch_id': 1, 'rx_decode_time_ms': 0.036716461181640625, 'rx_transfer_bytes': 339, 'rx_in_flight_time_ms': 0.0}
2025-12-16 01:24:26,527 [PID:17815] DEBUG: [Orchestrator] Stage-0 completed request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing
2025-12-16 01:24:26,527 [PID:17815] DEBUG: [Orchestrator] Request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 finalized at stage-0
2025-12-16 01:24:26,780 [PID:17815] DEBUG: [Orchestrator] Forwarded request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-1
2025-12-16 01:24:44,789 [PID:17815] INFO: [StageMetrics] stage=1 req=0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num_tokens_out': 170, 'stage_gen_time_ms': 17991.965770721436, 'batch_id': 1, 'rx_decode_time_ms': 5.737543106079102, 'rx_transfer_bytes': 3148794, 'rx_in_flight_time_ms': 1.1227130889892578}
2025-12-16 01:24:44,789 [PID:17815] DEBUG: [Orchestrator] Stage-1 completed request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing
2025-12-16 01:24:44,790 [PID:17815] DEBUG: [Orchestrator] Forwarded request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-2
2025-12-16 01:24:44,914 [PID:17815] INFO: [StageMetrics] stage=2 req=0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num_tokens_out': 0, 'stage_gen_time_ms': 117.71297454833984, 'batch_id': 1, 'rx_decode_time_ms': 0.43487548828125, 'rx_transfer_bytes': 8393, 'rx_in_flight_time_ms': 0.5235671997070312}
2025-12-16 01:24:44,915 [PID:17815] DEBUG: [Orchestrator] Stage-2 completed request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing
2025-12-16 01:24:44,915 [PID:17815] DEBUG: [Orchestrator] Request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 finalized at stage-2
2025-12-16 01:24:44,915 [PID:17815] DEBUG: [Orchestrator] Request 0_b3b2dcb1-4c75-42de-a073-dcef52b9e557 fully completed (1/1)
2025-12-16 01:24:44,915 [PID:17815] DEBUG: [Orchestrator] All requests completed
2025-12-16 01:24:44,915 [PID:17815] INFO: [Summary] {'e2e_requests': 1, 'e2e_total_time_ms': 21893.684148788452, 'e2e_sum_time_ms': 21892.935752868652, 'e2e_total_tokens': 0, 'e2e_avg_time_per_request_ms': 21892.935752868652, 'e2e_avg_tokens_per_s': 0.0, 'wall_time_ms': 21893.684148788452, 'final_stage_id': 2, 'stages': [{'stage_id': 0, 'requests': 1, 'tokens': 52, 'total_time_ms': 3505.100727081299, 'avg_time_per_request_ms': 3505.100727081299, 'avg_tokens_per_s': 14.835522299897058}, {'stage_id': 1, 'requests': 1, 'tokens': 170, 'total_time_ms': 18008.86106491089, 'avg_time_per_request_ms': 18008.86106491089, 'avg_tokens_per_s': 9.43979740791238}, {'stage_id': 2, 'requests': 1, 'tokens': 0, 'total_time_ms': 124.7246265411377, 'avg_time_per_request_ms': 124.7246265411377, 'avg_tokens_per_s': 0.0}], 'transfers': [{'from_stage': 0, 'to_stage': 1, 'samples': 1, 'total_bytes': 3148794, 'total_time_ms': 5.67626953125, 'tx_mbps': 4437.835776, 'rx_samples': 1, 'rx_total_bytes': 3148794, 'rx_total_time_ms': 5.737543106079102, 'rx_mbps': 4390.442308539705, 'total_samples': 1, 'total_transfer_time_ms': 12.53652572631836, 'total_mbps': 2009.3567029593396}, {'from_stage': 1, 'to_stage': 2, 'samples': 1, 'total_bytes': 8393, 'total_time_ms': 0.35572052001953125, 'tx_mbps': 188.7549247828418, 'rx_samples': 1, 'rx_total_bytes': 8393, 'rx_total_time_ms': 0.43487548828125, 'rx_mbps': 154.39821698245615, 'total_samples': 1, 'total_transfer_time_ms': 1.3141632080078125, 'total_mbps': 51.092588493468796}]}

```

## If you do not need to print the vLLM features, you can run the script directly, or unset VLLM_LOGGING_LEVEL. 
```bash
unset VLLM_LOGGING_LEVEL
```

