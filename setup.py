from setuptools import setup, find_packages

setup(
    name="medical-memorization-study",
    version="1.0.0",
    description="Privacy Vulnerabilities in Medical Foundation Models",
    author="Anonymous",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "timm>=0.9.0",
        "pillow>=9.5.0",
        "scikit-learn>=1.2.0",
        "tensorboard>=2.13.0",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.8",
)
