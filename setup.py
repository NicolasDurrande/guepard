"""The setup script."""

from setuptools import find_namespace_packages, setup

test_requirements = ["pytest"]

setup(
    author="Nicolas Durrande",
    author_email="nicolas@shiftlab.ai",
    python_requires='>=3.7',
    description="Gaussian Process Model Aggregation using Implicit Observations",
    license="Apache 2",
    include_package_data=True,
    name='guepard',
    packages=find_namespace_packages(include=["guepard*"]),
    test_suite='tests',
    tests_require=test_requirements,
    version='0.1.0',
)
