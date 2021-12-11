from setuptools import setup, find_packages

setup(
        name='seq_graph_retro',
        version='1.0',
        description='Sequential graph edit model for retrosynthesis.',
        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
)
