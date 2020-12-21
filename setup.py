from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

DEBUG_FLAGS = ['--debug', '--generate-line-info'] # --device-debug
MACROS = [('BLOCKSIZE', '16')]

setup(name='bitgridsample',
    version='1.0.0',
    description='Like torch grid_sample, but for binary input. Each bit of an uint8 input pixel corresponds to a channel.',
    author='Thomas Pönitz',
    author_email='tasptz@gmail.com',
    package_dir={'': 'py'},
    packages=['bitgridsample'],
    ext_modules=[
        CUDAExtension('bitgridsample.bitgridsample_cuda',
            sources=[
                'src/bitgridsample_cuda.cpp',
                'src/bitgridsample_cuda_kernel.cu'
            ],
            define_macros=MACROS,
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-D' + '='.join(a) for a in MACROS] # + DEBUG_FLAGS
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires='torch>=1.6',
    setup_requires=['torch>=1.6', 'numpy>=1.19', 'pytest>=5.4'],
    python_requires='>=3.6'
)
