import os
import pathlib

from setuptools import setup


def package_files(directory: str, ignore_ext: list = []) -> list:
    """Returns the filepath for all files in a directory. Borrowed from https://stackoverflow.com/a/36693250"""
    paths = []
    ignore_ext = [ext.lower() for ext in ignore_ext]
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(path, filename)
            if pathlib.Path(filepath).suffix.lower().strip(".") in ignore_ext:
                continue
            paths.append(filepath)
    return paths


urdf_files = package_files("jkinpylib/urdfs/")
for file in urdf_files:
    assert os.path.isfile(file), f"Error: parsed filepath '{file}' does not exist"

urdf_files = [
    fname.replace("jkinpylib/", "") for fname in urdf_files
]  # filenames are relative to the root directory, but we want them relative to the root/jkinpylib directory
assert len(urdf_files) > 0, "No URDF files found"

for file in urdf_files:
    reconstructed = "jkinpylib/" + file
    assert os.path.isfile(reconstructed), f"Error: reconstructed filepath '{reconstructed}' does not exist"

setup(
    name="jkinpylib",
    version="0.0.5",
    author="Jeremy Morgan",
    author_email="jsmorgan6@gmail.com",
    scripts=[],
    url="https://github.com/jstmn/jkinpylib",
    license="LICENSE.txt",
    description="Jeremy's Kinematics Python Library",
    py_modules=[],
    long_description=open("README.md").read(),
    install_requires=["klampt", "numpy", "torch", "kinpy", "more_itertools", "roma", "tqdm"],
    extras_require={"dev": ["black", "pylint", "PyQt5"]},
    include_package_data=True,
    packages=["jkinpylib"],
    package_data={"jkinpylib": urdf_files},
)
