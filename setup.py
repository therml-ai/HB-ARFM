from setuptools import find_packages, setup

setup(
    name="bubblefusion",
    version="0.0.1",
    description="Train Diffusion Models on BubbleML",
    author="Xianwei Zou, Sheikh Md Shakeel Hassan, Arthur Feeney, Aparna Chandramowlishwaran",
    packages=find_packages(include=["bubblefusion", "bubblefusion.*"]),
    install_requires=[],  # keep torch & heavy deps managed by conda
    python_requires=">=3.10",
    include_package_data=True,
)
