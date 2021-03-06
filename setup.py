import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthdnm",
    version="0.1.3",
    author="James Guevara",
    author_email="guevara.james@gmail.com",
    description="De novo caller and training pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/james-guevara/synthdnm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'synthdnm = synthdnm.run:run_synthdnm',
            'synthdnm-build = synthdnm.create_synthetic_dnms:build_synthdnm',
        ],
    },
)
