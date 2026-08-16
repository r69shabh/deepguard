from setuptools import setup, find_packages

setup(
    name="anomaly-ids",
    version="1.0.0",
    description="Adaptive Cyber-Physical Security — Anomaly-Based Intrusion Detection",
    author="Rishabh Gusain",
    packages=find_packages(include=["anomaly_ids", "anomaly_ids.*", "pipelines", "pipelines.*"]),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "scikit-learn>=1.4.2",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "scipy>=1.13.0",
        "joblib>=1.4.0",
        "tqdm>=4.66.2",
        "tensorflow>=2.14.0",
        "pyyaml>=6.0",
        "pytest>=7.4.0",
    ],
    entry_points={
        "console_scripts": [
            "anomaly-ids=main:main",
        ],
    },
    python_requires=">=3.9",
)
