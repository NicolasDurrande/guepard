"""The setup script."""

from setuptools import find_namespace_packages, setup

requirements = [
    "tensorflow>=2.4.0; platform_system!='Darwin' or platform_machine!='arm64'",
    # NOTE: Support of Apple Silicon MacOS platforms is in an experimental mode
    "tensorflow-macos>=2.4.0; platform_system=='Darwin' and platform_machine=='arm64'",
]

test_requirements = ["pytest"]

setup(
    author="Nicolas Durrande",
    author_email="nicolas@shiftlab.ai",
    python_requires='>=3.8',
    description="Gaussian Process Model Aggregation using Implicit Observations",
    license="Apache 2",
    include_package_data=True,
    name='guepard',
    packages=find_namespace_packages(include=["guepard*"]),
    install_requires=requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version='0.1.0',
)
