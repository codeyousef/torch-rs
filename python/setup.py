from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="torch-rs",
    version="0.21.0",
    description="Python bindings for torch-rs - PyTorch in Rust",
    author="torch-rs contributors",
    rust_extensions=[
        RustExtension(
            "torch_rs",
            binding=Binding.PyO3,
            features=["pyo3/extension-module"],
            debug=False,
        ),
    ],
    packages=["torch_rs"],
    package_dir={"torch_rs": "torch_rs"},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
    ],
)