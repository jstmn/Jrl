from setuptools import setup

setup(
    name="jkinpylib",
    version="0.0.0",
    author="Jeremy Morgan",
    author_email="jsmorgan6@gmail.com",
    packages=[],
    scripts=[],
    url="http://pypi.python.org/pypi/jkinpylib/",
    license="LICENSE.txt",
    description="Jeremy's Kinematics Python Library",
    long_description=open("README.md").read(),
    install_requires=["klampt", "numpy", "torch", "black"],
    include_package_data=True,
    # package_data={"": ["model_descriptions.yaml"]},
)