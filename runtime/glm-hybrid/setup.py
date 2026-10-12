from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="glm_hybrid",
    ext_modules=[
        CUDAExtension(
            "glm_hybrid",
            ["csrc/bindings.cpp", "csrc/moe_mul1.cpp", "csrc/hybrid_bridge.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fno-strict-aliasing"],
                "nvcc": ["-O3", "-std=c++17", "-gencode=arch=compute_86,code=sm_86"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
