# Third-party notices

This source repository does not contain model weights.

- `runtime/vllm-overlay/` is derived from vLLM and from
  [spele1100/vllm-glm53](https://github.com/spele1100/vllm-glm53) (GLM-5.3 support for Ada/Ampere), both under the
  Apache License 2.0; see [`LICENSE`](LICENSE). It also carries files of the vendor vLLM build pinned in
  `repro.lock.json`, which is distributed under the same license.
- `runtime/exllamav3-overlay/` modifies [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  (Copyright turboderp), which is distributed under the MIT License. Its upstream license text applies to those
  files.
- `runtime/glm-hybrid/`, `runtime/glm53_host.py`, the scripts, configs, tests and documentation are original to
  this repository and distributed under the Apache License 2.0.
- The served model is [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (Copyright Z.AI,
  MIT License), quantized by turboderp as
  [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) (MIT License). Neither is
  redistributed here: `scripts/download_model.sh` fetches the checkpoint from its pinned upstream revision.
- PyTorch, CUDA, Triton, TileLang, xgrammar, Docker and Hugging Face tooling are external dependencies and are not
  redistributed by this source repository.

Users are responsible for reviewing the current upstream terms before publishing or using the checkpoint.
