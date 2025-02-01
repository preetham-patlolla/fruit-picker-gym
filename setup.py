import setuptools
from pathlib import Path

setuptools.setup(
    name='fruit_picker_gym',
    version='0.0.1',
    description="A OpenAI Gym Env for fruit picking robot",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="fruit_picker_gym*"),
    install_requires=['gym']  # And any other dependencies foo needs
)