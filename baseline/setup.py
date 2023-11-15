from setuptools import setup, find_packages

setup(
    name="ransac",
    version="0.1.0",
    description="Baseline package for line and circle detection with ransac",
    author="Syrine Kalleli",
    author_email="cyrine.kalleli@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
)
