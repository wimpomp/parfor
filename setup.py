import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parfor",
    version="2022.5.0",
    author="Wim Pomp",
    author_email="wimpomp@gmail.com",
    description="A package to mimic the use of parfor as done in Matlab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wimpomp/parfor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['tqdm>=4.50.0', 'dill>=0.3.0', 'psutil'],
)
