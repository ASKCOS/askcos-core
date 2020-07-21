# askcos-core
[![askcos-base](https://img.shields.io/badge/-askcos--base-lightgray?style=flat-square)](https://gitlab.com/mlpds_mit/ASKCOS/askcos-base)
[![askcos-data](https://img.shields.io/badge/-askcos--data-lightgray?style=flat-square)](https://gitlab.com/mlpds_mit/ASKCOS/askcos-data)
[![askcos-core](https://img.shields.io/badge/-askcos--core-blue?style=flat-square)](https://gitlab.com/mlpds_mit/ASKCOS/askcos-core)
[![askcos-site](https://img.shields.io/badge/-askcos--site-lightgray?style=flat-square)](https://gitlab.com/mlpds_mit/ASKCOS/askcos-site)
[![askcos-deploy](https://img.shields.io/badge/-askcos--deploy-lightgray?style=flat-square)](https://gitlab.com/mlpds_mit/ASKCOS/askcos-deploy)

Python package for the prediction of feasible synthetic routes towards a desired compound and associated tasks related to synthesis planning. Originally developed under the DARPA Make-It program and now being developed under the [MLPDS Consortium](http://mlpds.mit.edu).

## 2020.07 Release Notes

User notes:

Developer notes:

Bug fixes:

For old release notes, see the [ASKCOS releases page](https://gitlab.com/mlpds_mit/ASKCOS/ASKCOS/-/releases).

## Getting Started

This package can be used on its own as a normal Python package without deploying the full askcos application. To do so, make sure that the project directory is on your `PYTHONPATH` and that the dependencies listed in `requirements.txt` are satisfied. The data and models must be downloaded separately from the [`askcos-data`](https://gitlab.com/mlpds_mit/ASKCOS/askcos-data) repository and placed in `askcos-core/askcos/data`.

### Downloading with GitLab Deploy Tokens

This repository can be downloaded using deploy tokens, which provide __read-only__ access to the source code and our container registry in GitLab. The deploy tokens can be found on the [MLPDS Member Resources ASKCOS Versions Page](https://mlpds.mit.edu/member-resources-releases-versions/). The only software prerequisites are git, docker, and docker-compose.

```bash
$ export DEPLOY_TOKEN_USERNAME=
$ export DEPLOY_TOKEN_PASSWORD=
$ git clone https://$DEPLOY_TOKEN_USERNAME:$DEPLOY_TOKEN_PASSWORD@gitlab.com/mlpds_mit/askcos/askcos-core.git
```

### Building a Docker Image

Before building the `askcos-site` image, you log in to the ASKCOS GitLab registry to download the `askcos-base` and `askcos-data` images which are dependencies (or build them yourself). Use the same deploy tokens as above to log in to the registry:

```bash
docker login registry.gitlab.com -u $DEPLOY_TOKEN_USERNAME -p $DEPLOY_TOKEN_PASSWORD
```

Then, the askcos-core image can be built using the Dockerfile in this repository.

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
