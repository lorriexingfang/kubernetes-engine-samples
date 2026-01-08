import os
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

tpu_engine_config = {
    "enforce_eager": True,
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "tensor_parallel_size": 4,  # Use all 4 TPU chips
}

llm_config = LLMConfig(
    accelerator_type="TPU-V5LITEPOD",
    model_loading_config={
        "model_id": "llama-3-8b",
        "model_source": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    engine_kwargs=tpu_engine_config,
    resources_per_bundle={"TPU": 4},  # Single host, 4 TPU chips
    runtime_env={
        "env_vars": {
            "JAX_PLATFORMS": "tpu",
            "TPU_BACKEND_TYPE": "jax",
            "HUGGING_FACE_HUB_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
        }
    },
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)