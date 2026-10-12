from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="glm_fast",
    ext_modules=[
        CUDAExtension(
            "glm_fast",
            ["csrc/fast_bindings.cpp", "csrc/glm_fast.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-gencode=arch=compute_86,code=sm_86"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
