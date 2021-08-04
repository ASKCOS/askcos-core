# askcos-core
[![askcos-base](https://img.shields.io/badge/-askcos--base-lightgray?style=flat-square)](https://github.com/ASKCOS/askcos-base)
[![askcos-data](https://img.shields.io/badge/-askcos--data-lightgray?style=flat-square)](https://github.com/ASKCOS/askcos-data)
[![askcos-core](https://img.shields.io/badge/-askcos--core-blue?style=flat-square)](https://github.com/ASKCOS/askcos-core)
[![askcos-site](https://img.shields.io/badge/-askcos--site-lightgray?style=flat-square)](https://github.com/ASKCOS/askcos-site)
[![askcos-deploy](https://img.shields.io/badge/-askcos--deploy-lightgray?style=flat-square)](https://github.com/ASKCOS/askcos-deploy)

Python package for the prediction of feasible synthetic routes towards a desired compound and associated tasks related to synthesis planning. Originally developed under the DARPA Make-It program and now being developed under the [MLPDS Consortium](http://mlpds.mit.edu).

## Getting Started

This package can be used on its own as a normal Python package without deploying the full ASKCOS application. To do so, make sure that the project directory is on your `PYTHONPATH` and that the dependencies listed in `requirements.txt` are satisfied. The data and models must be downloaded separately from the [`askcos-data`](https://github.com/ASKCOS/askcos-data) repository and placed in `askcos-core/askcos/data`.

### Building a Docker Image

The `askcos-core` image can be built using the Dockerfile in this repository. It depends on the `askcos-data` Docker image, which can be built manually or pulled from Docker Hub.

```bash
$ cd askcos-core
$ docker build -t <image name> .
```

A Makefile is also provided to simplify the build command by providing a default image name and tag:

```bash
$ cd askcos-core
$ make build
```

### How To Run Individual Modules
Many of the individual modules -- at least the ones that are the most interesting -- can be run "standalone". Examples of how to use them are often found in the ```if __name__ == '__main__'``` statement at the bottom of the script definitions. For example...

Using the learned synthetic complexity metric (SCScore):
```
askcos/prioritization/precursors/scscore.py
```

Obtaining a single-step retrosynthetic suggestion with consideration of chirality:
```
askcos/retrosynthetic/transformer.py
```

Finding recommended reaction conditions based on a trained neural network model:
```
askcos/synthetic/context/neuralnetwork.py
```

Using the template-free forward predictor:
```
askcos/synthetic/evaluation/template_free.py
```

Using the coarse "fast filter" (binary classifier) for evaluating reaction plausibility:
```
askcos/synthetic/evaluation/fast_filter.py
```
