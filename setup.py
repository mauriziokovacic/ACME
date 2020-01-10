import setuptools

__version__ = '1.0.0'
url = 'https://github.com/mauriziokovacic/ACME'

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'numpy',
    'torch',
    'torch_scatter',
    'torch_geometric',
    'neural_renderer_pytorch',
]

setuptools.setup(
    name="ACME",
    version=__version__,
    author="Maurizio Kovacic",
    author_email="maurizio.kovacic@gmail.com",
    description="A Python Library containing several algorithms and utilities for Python, Pytorch, geometry processing, and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="url",
    keywords=[
        'python',
        'pytorch',
        'geometry-processing',
        'animation',
        'machine-learning',
    ],
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
