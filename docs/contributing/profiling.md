# Profiling vLLM-Omni

This document provides a **practical, example-driven guide** to the logging and profiling system in **vLLM-Omni**. Instead of only describing what is logged, it explains **what to run**, **what logs you will see**, and **how to interpret them**.vLLM-Omni currently supports **two fundamentally different inference scheduling paths**, and the available profiling signals differ accordingly.

• **Diffusion/DiT Single diffusion**

**Examples:**
- `examples/offline_inference/text_to_image`[[text_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image)
- `examples/offline_inference/image_to_image`[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)
- `examples/offline_inference/text_to_video`[[text_to_video]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video)

• **Characteristics**:

- Single diffusion pipeline
- No Omni stage scheduler involved
- Logs mainly come from **vLLM core** and **diffusion engine**


• **Multi-Stage Pipeline**

**Examples:**

- `examples/offline_inference/qwen2_5_omni`[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)
- `examples/offline_inference/qwen3_omni`[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)

Characteristics:

- Multi-process, multi-stage scheduling
- Explicit Omni stage orchestration
- Logs include **vLLM core + Omni + diffusion + IPC**

The logging content and usage methods of the logger system under different scheduling paths are as follows:
## Recording Content and Usage Instructions
### 1. VLLM features(All Pipelines)
VLLM features it log for root module vllm, and the sub model automatically inherit the parent logger. But the vllm_omni module failed to automatically inherit vllm.So we need to init vllm_omni root logger, witch inherit the parent logger.vLLM config includes communication methods, scheduling modes, parallelism, and runtime scale. It also includes shared memory pressure status, model size, and observed GPU memory usage during runtime.The VLLM config content recorded by Single the Diffusion Pipeline model and the Multi-Stage Pipeline model is the same.
#### How to view vllm features
Before running the scripts in the examples, set the environment variables to view the vLLM config in the logs printed in the terminal.
 ```bash
 export VLLM_LOGGING_LEVEL=DEBUG
 ```
#### Examples Log Output
 ```json
INFO 12-22 03:08:19 [vllm_omni/diffusion/diffusion_engine.py:92] Starting server...
DEBUG 12-22 03:08:22 [plugins/__init__.py:28] No plugins for group vllm.platform_plugins found.
DEBUG 12-22 03:08:22 [platforms/__init__.py:34] Checking if TPU platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 12-22 03:08:22 [platforms/__init__.py:58] Checking if CUDA platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:78] Confirmed CUDA platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:106] Checking if ROCm platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 12-22 03:08:22 [platforms/__init__.py:127] Checking if XPU platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 12-22 03:08:22 [platforms/__init__.py:153] Checking if CPU platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:58] Checking if CUDA platform is available.
DEBUG 12-22 03:08:22 [platforms/__init__.py:78] Confirmed CUDA platform is available.
INFO 12-22 03:08:22 [platforms/__init__.py:216] Automatically detected platform cuda.
DEBUG 12-22 03:08:27 [compilation/decorators.py:155] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.qwen3_moe.Qwen3MoeModel'>: ['input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds']
 ```
#### Analysis of Sample Log Output
| Log Stage | vLLM Feature |Values |
|----------|-------------|------------|
| Engine startup | Engine lifecycle management | Confirms the runtime enters the vLLM execution path successfully |
| Plugin system | Extensible plugin architecture |  confirms no custom platform extensions are enabled |
| Platform probing | Automatic multi-backend detection | Eliminates the need for manual backend configuration |
| Accelerator probing | TPU/XPU/CUDA/ROCm  backend support | Specify the type of accelerator |

This table shows how vLLM startup logs expose backend detection, execution platform selection, dynamic shape support, and model compatibility, enabling users to quickly validate environment correctness and runtime behavior.

### 2.vLLM-omni features
The vllm-omni feature provides multi-dimensional metrics such as end-to-end performance, IPC communication, pipeline scheduling, and engine passthrough, enabling full observability and detailed performance analysis throughout the entire multimodal inference process. However, since the Diffusion Pipeline model does not schedule the omni feature, only the Multi-Stage Pipeline model can access the omni feature.[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)
#### How to view vLLM-omni features
During the operation of the Multi-Stage Pipeline model, the Omni feature is automatically invoked. You can directly run the script to view the Omni feature of the model.[[qwen2_5_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen2_5_omni)[[qwen3_omni]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_omni)
```bash
sh run_multiple_prompts.sh
```
#### Examples Log Output
 ```json
{
  "e2e_requests": 1,
  "e2e_total_time_ms": 23745.417833328247,
  "e2e_sum_time_ms": 23744.66896057129,
  "e2e_total_tokens": 0,
  "e2e_avg_time_per_request_ms": 23744.66896057129,
  "e2e_avg_tokens_per_s": 0.0,
  "wall_time_ms": 23745.417833328247,
  "final_stage_id": 2,
  "stages": [
    {
      "stage_id": 0,
      "requests": 1,
      "tokens": 62,
      "total_time_ms": 1299.1509437561035,
      "avg_time_per_request_ms": 1299.1509437561035,
      "avg_tokens_per_s": 47.72347685846702
    },
    {
      "stage_id": 1,
      "requests": 1,
      "tokens": 976,
      "total_time_ms": 17227.604389190674,
      "avg_time_per_request_ms": 17227.604389190674,
      "avg_tokens_per_s": 56.6532628652875
    },
    {
      "stage_id": 2,
      "requests": 1,
      "tokens": 0,
      "total_time_ms": 5190.91010093689,
      "avg_time_per_request_ms": 5190.91010093689,
      "avg_tokens_per_s": 0.0
    }
  ],
  "transfers": [
    {
      "from_stage": 0,
      "to_stage": 1,
      "samples": 1,
      "total_bytes": 1779789,
      "total_time_ms": 3.548145294189453,
      "tx_mbps": 4012.888655748421,
      "rx_samples": 1,
      "rx_total_bytes": 1779789,
      "rx_total_time_ms": 2.7894973754882812,
      "rx_mbps": 5104.257177337436,
      "total_samples": 1,
      "total_transfer_time_ms": 7.725238800048828,
      "total_mbps": 1843.0902097045862
    },
    {
      "from_stage": 1,
      "to_stage": 2,
      "samples": 1,
      "total_bytes": 3572,
      "total_time_ms": 0.42510032653808594,
      "tx_mbps": 67.22177852159282,
      "rx_samples": 1,
      "rx_total_bytes": 3572,
      "rx_total_time_ms": 0.3783702850341797,
      "rx_mbps": 75.52390113673599,
      "total_samples": 1,
      "total_transfer_time_ms": 1.4834403991699219,
      "total_mbps": 19.26332868916747
    }
  ]
}

 ```

#### Analysis of Sample Log Output
| Category    | Metric / Component                                                                               | What It Represents                 | User Value                       |
| ----------- |--------------------------------------------------------------------------------------------------|------------------------------------| -------------------------------- |
| End-to-End  | `e2e_requests`;`e2e_total_time_ms`;`e2e_avg_time_per_request_ms`;`wall_time_ms`;`final_stage_id` | Total completed requests(latency;Wall-clock runtime) | Confirms full pipeline execution |
| IPC / Transfer | Stage-0 → Stage-1                                                                                | High-volume tensor transfer        | Validates high-bandwidth IPC     |
| Observability | IPC visibility;E2E aggregation                                                                                   | Per-stage time & token breakdown   | Precise bottleneck isolation     |

The Omni summary provides a unified view of end-to-end latency, stage-level compute costs, and inter-stage communication metrics, enabling precise performance diagnosis in multi-stage pipelines.



### 3.Diffusion feature
• The Multi-Stage Pipeline logs do not directly record the details of the diffusion algorithm. Instead, they abstract a complete diffusion process into a single Stage, indirectly reflecting the overall performance of diffusion through `stage_gen_time_ms`, and focus on recording IPC and scheduling characteristics across different Stages.

• The Diffusion Pipeline logs comprehensively cover the core macro characteristics of diffusion inference, including model loading, CFG, number of inference steps, total diffusion time, average denoising step time, and other parameters.



#### How to view Diffusion features
1.The Multi-Stage Pipeline

##### Setting the log switch:

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
##### Setting the log switch:

```bash
sh run_multiple_prompts.sh
```
#### Examples Log Output
 ```json
2025-12-22 03:43:59,499 [PID:28962] INFO: [StageMetrics] stage=0 req=0_6850c39c-0977-479d-8b83-98a582b55e36 metrics={'num_tokens_out': 62, 'stage_gen_time_ms': 1285.4697704315186, 'batch_id': 1, 'rx_decode_time_ms': 0.020742416381835938, 'rx_transfer_bytes': 339, 'rx_in_flight_time_ms': 0.0}
2025-12-22 03:43:59,499 [PID:28962] DEBUG: [Orchestrator] Stage-0 completed request 0_6850c39c-0977-479d-8b83-98a582b55e36; forwarding or finalizing
2025-12-22 03:43:59,500 [PID:28962] DEBUG: [Orchestrator] Request 0_6850c39c-0977-479d-8b83-98a582b55e36 finalized at stage-0
2025-12-22 03:43:59,525 [PID:28962] DEBUG: [Orchestrator] Forwarded request 0_6850c39c-0977-479d-8b83-98a582b55e36 to stage-1
2025-12-22 03:44:16,752 [PID:28962] INFO: [StageMetrics] stage=1 req=0_6850c39c-0977-479d-8b83-98a582b55e36 metrics={'num_tokens_out': 976, 'stage_gen_time_ms': 17209.781646728516, 'batch_id': 1, 'rx_decode_time_ms': 2.7894973754882812, 'rx_transfer_bytes': 1779789, 'rx_in_flight_time_ms': 1.3875961303710938}
2025-12-22 03:44:16,753 [PID:28962] DEBUG: [Orchestrator] Stage-1 completed request 0_6850c39c-0977-479d-8b83-98a582b55e36; forwarding or finalizing
2025-12-22 03:44:16,754 [PID:28962] DEBUG: [Orchestrator] Forwarded request 0_6850c39c-0977-479d-8b83-98a582b55e36 to stage-2
2025-12-22 03:44:21,943 [PID:28962] INFO: [StageMetrics] stage=2 req=0_6850c39c-0977-479d-8b83-98a582b55e36 metrics={'num_tokens_out': 0, 'stage_gen_time_ms': 5178.144931793213, 'batch_id': 1, 'rx_decode_time_ms': 0.3783702850341797, 'rx_transfer_bytes': 3572, 'rx_in_flight_time_ms': 0.6799697875976562}
2025-12-22 03:44:21,944 [PID:28962] DEBUG: [Orchestrator] Stage-2 completed request 0_6850c39c-0977-479d-8b83-98a582b55e36; forwarding or finalizing
 ```
#### Analysis of Sample Log Output
| Stage ID | Tokens Out | Stage Time (ms) | RX Transfer (Bytes) | RX Decode Time (ms) | RX In-Flight Time (ms) | Stage Role             | User Insight                |
| -------- | ---------- | --------------- | ------------------- | ------------------- | ---------------------- | ---------------------- | --------------------------- |
| Stage-0  | 62         | 1285.47         | 339                 | 0.02                | 0.00                   | Preprocessing / Encoding | Low latency, minimal IPC cost |
| Stage-1  | 976        | 17209.78        | 1 779 789           | 2.79                | 1.39                   | Primary compute stage* | Dominant latency contributor|
| Stage-2  | 0          | 5178.14         | 3572                | 0.38                | 0.68                   | Post-processing / Decoding | Non-token workload visibility |

StageMetrics logs provide per-stage latency, token output, and IPC statistics, enabling precise identification of compute-heavy stages and communication overhead in vLLM-Omni’s multi-stage pipeline.

2.The Diffusion Pipeline

Run the Diffusion Pipeline script directly to view the model's diffusion properties(Taking image_to_image as an example, the usage method for other models is the same.)[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image)[[image_to_image]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image)[[text_to_video]](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video):

```python
python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0

```
