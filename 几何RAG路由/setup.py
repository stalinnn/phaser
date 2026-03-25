from setuptools import setup, find_packages

setup(
    name="geom_rag_router",
    version="0.1.0",
    description="Geometric RAG Router: Automatically route RAG queries based on Data Manifold Curvature (Gromov Delta-Hyperbolicity).",
    author="Unified Field AI",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
