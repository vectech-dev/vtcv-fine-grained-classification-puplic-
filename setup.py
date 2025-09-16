from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"



setup(
    name="vtcv-fine-grained-classification",
    version=__version__,
    description="improving self assessment classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sameerah Talafha, Thomas Jenkins",
    author_email="sameerah@vectech.io",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "timm",
        "tensorboard",
        "seaborn",
        "IPython",
        "boto3",
        "black",
        "isort",
        "albumentations",
        "torchsampler",
        "pretrainedmodels",
        "fastai",
        "torchsummary",
        "torch >= 1.4",
        "torchvision",
        "Ranger21  @ git+https://github.com/lessw2020/Ranger21.git",
        # "vtcv-core-classification @ git+ssh://git@github.com/vectech-dev/vtcv-core-classification.git",
        "vtcv-core-classification @ git+ssh://git@github.com/vectech-dev/vtcv-core-classification.git@7b4fdb69849c12c58f93d504edb9dbd770005070",
        "vtcv-image-transforms @ git+ssh://git@github.com/vectech-dev/vtcv-image-transforms.git",
        "mosmask-unet @ git+ssh://git@github.com/vectech-dev/mosmask-unet.git",
        
    ],
   
    entry_points={
        "console_scripts": [
            "vtcv-fine-grained-classification = vtcv_fine_grained_classification.__main__:run"
        ]
    },
)
