from setuptools import setup, find_packages

setup(
    name="semiconductor-yield-intel",
    version="0.1.0",
    description="GNN-based wafer yield prediction with causal root cause attribution and active learning",
    author="Shruti Dhamdhere",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
