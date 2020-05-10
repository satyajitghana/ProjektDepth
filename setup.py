import setuptools

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="vathos",  # Replace with your own username
    version="0.0.1",
    author="Satyajit Ghana",
    author_email="satyajitghana7@gmail.com",
    description="ProjektDepth_Vathos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/satyajitghana/ProjektDepth",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=required,
    include_package_data=True,
)
