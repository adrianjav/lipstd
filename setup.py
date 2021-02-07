# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='lipstd',
     version='0.0.1',
     scripts=[],
     author="Adrián Javaloy",
     author_email="adrian.javaloy@gmail.com",
     description="WIP",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/adrianjav/lipstd",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.7",
         # "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     include_package_data=True,
     install_requires=[
         'torch',  # WIP
     ]
)