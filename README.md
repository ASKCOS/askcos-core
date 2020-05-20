# ASKCOS
Software package for the prediction of feasible synthetic routes towards a desired compound and associated tasks related to synthesis planning. Originally developed under the DARPA Make-It program and now being developed under the [MLPDS Consortium](http://mlpds.mit.edu).

# Contents
* [2020.04 Release](#202004-release)
    * [Release Notes](#release-notes)
    * [Using GitLab Deploy Tokens](#using-gitlab-deploy-tokens)
    * [Upgrade Information](#upgrade-information)
* [First Time Deployment with Docker](#first-time-deployment-with-docker)
    * [Prerequisites](#prerequisites)
    * [Deploying the Web Application](#deploying-the-web-application)
    * [Backing Up User Data](#backing-up-user-data)
    * [(Optional) Building the ASKCOS Image](#optional-building-the-askcos-image)
    * [Add Customization](#add-customization)
    * [Managing Django](#managing-django)
    * [Important Notes](#important-notes)
        * [Scaling Workers](#scaling-workers)
* [How To Run Individual Modules](#how-to-run-individual-modules)

# 2020.04 Release

## Release Notes

User notes:  
* New general selectivity model available from the Interactive Path Planner UI and as an API endpoint (MR !319).
* Redesigned and consolidated forward prediction UI combining reaction condition prediction, forward synthesis prediction, and impurity prediction (MR !279).
* Drawing tool added to Interactive Path Planner UI (MR !282).
* The Interactive Path Planner now saves last used settings locally in the browser, and more visualization settings are exposed to the user (MR !306).
* Users can now initiate a tree builder search from the Interactive Path Planner (Issue #285; MR !299). Users can now initiate a tree builder job by clicking a new button on the Interactive Path Planner, rather than having to go to the tree builder page. Once the target smiles string has been entered into the Interactive Path Planner, simply click the “Build tree” button to start an asynchronous tree builder job. The user will receive a notification when the job(s) finish so they can view the results in a new tab.
* Option added to automatically redirect to the Interactive Path Planner UI upon completion of tree builder (using new UI) (Issue #269).
* Show building block source in the Interactive Path Planner and tree visualization UIs (Issue #270; MR !301).
* Reaction precedents for new template sets can be viewed in the UI (MR !295).
* Redesigned page for viewing and adding banned chemicals and reactions (MR !305).

Developer notes:
* Enabled versioning for the API (Issue #278; MR !296). This allows new functionality to be added which could break compatibility with previous API’s. For example, requiring everything to be a POST request. POST requests accept JSON data in the request body and make it easier to handle lists and boolean values.
* The deploy folder has been migrated to its own repository `askcos-deploy` (Issue #261). It now has no inter-dependencies with the data or models. For basic deployment, cloning `askcos-deploy` is now sufficient.
* API endpoint created for the atom mapping tool (Issue #268).
* API endpoint created for the Impurity Predictor (Issue #256). Examples were included in the corresponding documentation.
* Three new scoring coordinator specific workers have been created to handle template-free prediction, template-based prediction and fast filter evaluation respectively. Coordination will be handled on the client rather than on the server (Issue #250).
* Reconfigure the Docker image so that new templates can be added without the image needing to be rebuilt (Issue #247; MR !298).
* Additional data can now be added/appended to collections in the mongodb via the deploy script without clearing the collection first (Issue #246).
* Chem Historian information has been migrated into the mongodb. This allows the data to be accessed via a db lookup and should reduce the applications memory footprint (Issue #205).
* Make it easier for companies/individuals to use their own retrained models and template sets (Issue #154).
* Use tokens to authenticate users that make API calls (Issue #107).
* Refactored MCTS code to decouple pure python code from celery infrastructure (MR !283).
* Added Makefile to facilitate building docker images (MR !285)
* Added option to retain atom mapping for forward prediction API calls (Issue #262; MR !308).


Bug fixes:
* Remove broken links from the old context pages (Issue #277).
* Forward Predictor API may return -Infinity in JSON response (Issue #275; MR !300).
* Running the main tree.builder.py file as a script or the tree builder unit test with multiprocessing did not work (Issue #274; MR !283).
* Atomic identity should not change in a tree builder reaction prediction (Issues #255, #266, #296; MRs !274, !314).


## Using GitLab Deploy Tokens

ASKCOS can also be downloaded using deploy tokens, these provide __read-only__ access to the source code and our container registry in GitLab. Below is a complete example showing how to deploy the ASKCOS application using deploy tokens (omitted in this example). The deploy tokens can be found on the [MLPDS Member Resources ASKCOS Versions Page](https://mlpds.mit.edu/member-resources-releases-versions/). The only software prerequisites are git, docker, and docker-compose.

```bash
$ export DEPLOY_TOKEN_USERNAME=
$ export DEPLOY_TOKEN_PASSWORD=
$ docker login registry.gitlab.com -u $DEPLOY_TOKEN_USERNAME -p $DEPLOY_TOKEN_PASSWORD
$ git clone https://$DEPLOY_TOKEN_USERNAME:$DEPLOY_TOKEN_PASSWORD@gitlab.com/mlpds_mit/askcos/askcos-deploy.git
$ cd askcos-deploy
$ git checkout 2020.04
$ bash deploy.sh deploy
```

## Upgrade Information

The easiest way to upgrade to a new version of ASKCOS is using Docker and docker-compose.
To get started, make sure both docker and docker-compose are installed on your machine.
We have a pre-built docker image of ASKCOS hosted on GitLab.
It is a private repository; if you do not have access to pull the image, please [contact us](mailto:mlpds_support@mit.edu).
Start with the `askcos-deploy` repository. The process for cloning the repository and checking out the correct version tag is described above.

### From v0.3.1 or above
```bash
$ git checkout 2020.04
$ bash deploy.sh update -v 2020.04
```

If you have not seeded the database before (if you're upgrading from v0.3.1), you will need to do so:
```bash
$ bash deploy.sh set-db-defaults seed-db
```

### From v0.2.x or v0.3.0
```bash
$ git checkout 2020.04
$ bash backup.sh
$ bash deploy.sh update -v 2020.04
$ bash deploy.sh set-db-defaults seed-db
$ bash restore.sh
```

__NOTE:__ A large amount of data has been migrated to the mongo db starting in v0.4.1 (chemhistorian), and seeding may take some time to complete. We send this seeding task to the background so the rest of the application can start and become functional without having to wait. If using the default set of data (i.e. - using the exact commands above), you can monitor the progress of mongo db seeding using `bash deploy.sh count-mongo-docs`, which will tell you how many documents have been seeded out of the expected number. Complete seeding is not necessary for application functionality unless you use the chemical popularity logic in the tree builder.

# First Time Deployment with Docker

## Prerequisites

 - If you're building the image from scratch, make sure git (and git lfs) is installed on your machine
 - Install Docker [OS specific instructions](https://docs.docker.com/install/)
 - Install docker-compose [installation instructions](https://docs.docker.com/compose/install/#install-compose)

## Deploying the Web Application

Deployment is initiated by a bash script that runs a few docker-compose commands in a specific order.
Several database services need to be started first, and more importantly seeded with data, before other services 
(which rely on the availability of data in the database) can start. The bash script can be found and should be run 
from the deploy folder as follows:

```bash
$ bash deploy.sh command [optional arguments]
```

There are a number of available commands, including the following for common deploy tasks:
* `deploy`: runs standard first-time deployment tasks, including `seed-db`
* `update`: pulls new docker image from GitLab repository and restarts all services
* `seed-db`: seed the database with default or custom data files
* `start`: start a deployment without performing first-time tasks
* `stop`: stop a running deployment
* `clean`: stop a running deployment and remove all docker containers

For a running deployment, new data can be seeded into the database using the `seed-db` command along with arguments
indicating the types of data to be seeded. Note that this will replace the existing data in the database.
The available arguments are as follows:
* `-b, --buyables`: specify buyables data to seed, either `default` or path to data file
* `-c, --chemicals`: specify chemicals data to seed, either `default` or path to data file
* `-x, --reactions`: specify reactions data to seed, either `default` or path to data file
* `-r, --retro-templates`: specify retrosynthetic templates to seed, either `default` or path to data file
* `-f, --forward-templates`: specify forward templates to seed, either `default` or path to data file

For example, to seed default buyables data and custom retrosynthetic pathways, run the following from the deploy folder:

```bash
$ bash deploy.sh seed-db --buyables default --retro-templates /path/to/my.retro.templates.json.gz
```

To update a deployment, run the following from the deploy folder:

```bash
$ bash deploy.sh update --version x.y.z
```

To stop a currently running application, run the following from the deploy folder:

```bash
$ bash deploy.sh stop
```

If you would like to clean up and remove everything from a previous deployment (__NOTE: you will lose user data__), run the following from the deploy folder:

```bash
$ bash deploy.sh clean
```

## Backing Up User Data

If you are upgrading from v0.3.1 or later, the backup/restore process is no longer needed unless you are moving deployments to a new machine.

If you are upgrading the deployment from a previous version (prior to v0.3.1), or moving the application to a different server, you may want to retain user accounts and user-saved data/results.
The provided `backup.sh` and `restore.sh` scripts are capable of handling the backup and restoring process. Please read the following carefully so as to not lose any user data:

1) Start by making sure the previous version you would like to backup is __currently up and running__ with `docker-compose ps`.
2) Checkout the newest version of the source code `git checkout 2020.04`
3) Run `$ bash backup.sh`
4) Make sure that the `deploy/backup` folder is present, and there is a folder with a long string of numbers (year+month+date+time) that corresponds to the time you just ran the backup command
5) If the backup was successful (`db.json` and `user_saves` (\<v0.3.1) or `results.mongo` (\>=0.3.1) should be present), you can safely tear down the old application with `docker-compose down [-v]`
6) Deploy the new application with `bash deploy.sh deploy` or update with `bash deploy.sh update -v x.y.z`
7) Restore user data with `bash restore.sh`

Note: For versions >=0.3.1, user data persists in docker volumes and is not tied to the lifecycle of the container services. If the [-v] flag is not used with `docker-compose down`, volumes do not get removed, and user data is safe. In this case, the backup/restore procedure is not necessary as the containers that get created upon an install/upgrade will continue to use the docker volumes that contain all the important data. If the [-v] flag is used, all data will be removed and a restore will be required to recover user data.

## (Optional) Building the ASKCOS Image

The askcos image itself can be built using the Dockerfile in this repository.

```bash
$ git clone https://gitlab.com/mlpds_mit/askcos/askcos  
$ cd askcos/  
$ git lfs pull   
$ docker build -t <image name> .
```

__NOTE:__ The image name should correspond with what exists in the `docker-compose.yml` file. By default, the image name is environment variable `ASKCOS_IMAGE_REGISTRY` + `askcos`. If you choose to use a custom image name, make sure to modify the `ASKCOS_IMAGE_REGISTRY` variable or the `docker-compose.yml` file accordingly.

__NOTE:__ When re-deploying the application after building a custom image with the same name/tag as one in our repository, you can supply the `--local` flag to the deployment script which will skip pulling the image from our container registry.

## Add Customization

There are a few parts of the application that you can customize:
* Header sub-title next to ASKCOS (to designate this as a local deployment at your organization)

This is handled as an environment variable that can change upon deployment (and are therefore not tied into the image directly). This can be found in `deploy/customization`. Please let us know what other degrees of customization you would like.

## Managing Django

If you'd like to manage the Django app (i.e. - run python manage.py ...), for example, to create an admin superuser, you can run commands in the _running_ app service (do this _after_ `docker-compose up`) as follows:

```bash
$ docker-compose exec app bash -c "python /usr/local/ASKCOS/askcos/manage.py createsuperuser"
```

In this case you'll be presented an interactive prompt to create a superuser with your desired credentials.

## Important Notes

### Scaling Workers

Only 1 worker per queue is deployed by default with limited concurrency. This is not ideal for many-user demand. 
The scaling of each worker is defined at the top of the `deploy.sh` script. 
To scale a desired worker, change the appropriate value in `deploy.sh`, for example:
```
n_tb_c_worker=N          # Tree builder chiral worker
```

where N is the number of workers you want. Then run `bash deploy.sh start [-v <version>]`.


# How To Run Individual Modules
Many of the individual modules -- at least the ones that are the most interesting -- can be run "standalone". Examples of how to use them are often found in the ```if __name__ == '__main__'``` statement at the bottom of the script definitions. For example...

Using the learned synthetic complexity metric (SCScore):
```
makeit/prioritization/precursors/scscore.py
```

Obtaining a single-step retrosynthetic suggestion with consideration of chirality:
```
makeit/retrosynthetic/transformer.py
```

Finding recommended reaction conditions based on a trained neural network model:
```
makeit/synthetic/context/neuralnetwork.py
```

Using the template-free forward predictor:
```
makeit/synthetic/evaluation/template_free.py
```

Using the coarse "fast filter" (binary classifier) for evaluating reaction plausibility:
```
makeit/synthetic/evaluation/fast_filter.py
```
