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
We can see the vLLM logs in the console and the diffusion logs in path/omni_diffusion_stats/omni_diffusion_%Y%m%d_%H%M%S_xx_pidxxxxW.jsonl.

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
## omni_diffusion_%Y%m%d_%H%M%S_xx_pidxxxxW.jsonl
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
INFO 12-17 09:22:16 [omni_diffusion.py:114] OmniDiffusion initialized: model=/cy50055764/cy50055764/models/Qwen-Image-Edit, class=QwenImageEditPipeline, init_ms=36702.19
Pipeline loaded

```
---

