import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'torch', 'torch_scatter','torch_geometric', 'neural_renderer']

setuptools.setup(
    name="ACME",
    version="1.0.0",
    author="Maurizio Kovacic",
    author_email="maurizio.kovacic@gmail.com",
    description="A Python Library that contains a bunch of useful algorithms and utilities for geometry processing and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mauriziokovacic/ACME",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
