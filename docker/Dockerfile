FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# The published measurements used transformers 5.16.1 / tokenizers 0.23.1; the base image ships 5.15.1 / 0.22.2.
RUN python3 -m pip install --no-cache-dir --no-deps transformers==5.16.1 tokenizers==0.23.1 huggingface-hub==1.29.0

# vLLM: GLM-5.3 support (spele1100/vllm-glm53) plus this project's changes, as a Python-only overlay. The base
# image's compiled extensions are byte-identical to those the runtime was measured with.
COPY runtime/install_overlay.py /opt/glm53/runtime/install_overlay.py
COPY runtime/vllm-overlay /opt/glm53/runtime/vllm-overlay
RUN python3 /opt/glm53/runtime/install_overlay.py /opt/glm53/runtime/vllm-overlay

# Extensions below compile against the base image's nvcc (CUDA 13.0) and the pip CUDA 13 headers.
ARG CUDA_PIP_INCLUDE=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include

# ExLlamaV3 at a pinned upstream commit plus this project's EXL3 MoE decode kernels, built for sm_86.
ARG EXLLAMAV3_REPO=https://github.com/turboderp-org/exllamav3.git
ARG EXLLAMAV3_COMMIT=499890c75d20d8e7c9d061f37189ae611a5c9f0b
ARG MAX_JOBS=8
COPY runtime/exllamav3-overlay /opt/glm53/runtime/exllamav3-overlay
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/* \
 && git clone --quiet --filter=blob:none "$EXLLAMAV3_REPO" /tmp/exllamav3 \
 && git -C /tmp/exllamav3 checkout --quiet "$EXLLAMAV3_COMMIT" \
 && python3 /opt/glm53/runtime/install_overlay.py /opt/glm53/runtime/exllamav3-overlay --destination /tmp/exllamav3 \
 && cd /tmp/exllamav3 \
 && CPATH="$CUDA_PIP_INCLUDE" TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS="$MAX_JOBS" python3 -m pip install --no-cache-dir --no-deps --no-build-isolation . \
 && cd / && rm -rf /tmp/exllamav3

# glm_hybrid (CPU expert pools + graph-safe CUDA bridge) and glm_fast (fused mHC / router / top-k kernels), sm_86.
COPY runtime/glm-hybrid /tmp/glm-hybrid
RUN cd /tmp/glm-hybrid \
 && export CPATH="$CUDA_PIP_INCLUDE" \
 && python3 setup.py build_ext --inplace && python3 setup_fast.py build_ext --inplace \
 && cp glm_hybrid*.so glm_fast*.so "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/" \
 && cd / && rm -rf /tmp/glm-hybrid \
 && python3 -c 'import torch, exllamav3_ext, glm_hybrid, glm_fast; assert hasattr(exllamav3_ext, "exl3_moe")'

COPY runtime/glm53_host.py /opt/glm53/runtime/glm53_host.py
COPY configs/2x3090-256k-agent.env configs/lru-plan-2x3090-13312m-k3.json configs/hot-expert-rankings.json \
     configs/chat-template-bounded.jinja /opt/glm53/configs/
COPY scripts/serve-container.sh /usr/local/bin/glm53-serve
RUN chmod 0755 /usr/local/bin/glm53-serve

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/glm53-serve"]
