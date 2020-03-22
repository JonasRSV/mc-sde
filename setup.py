import setuptools

setuptools.setup(
    name="sdepy",
    version="0.0.1",
    author="Jonas",
    author_email="jonas@valfridsson.net",
    description="",
    url="",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=["numpy", "mpi4py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
