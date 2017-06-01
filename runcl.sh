#!/bin/bash
# Wrapper script for running CodaLab jobs.
#
# Usage:
#
#   ./runcl.sh [--gpu] [--resume UUID EPOCH] [--host NODENAME] [ARGS [ARGS ...]]
#
# --gpu                 Use GPU
# --resume UUID EPOCH   Resume training on the run with UUID from the specified EPOCH.
# --host NODENAME       (codalab.stanford.edu only) Issue job to host with name NODENAME.
# ARGS [ARGS ...]       Arguments to pass onto basic_model.py, like "--learning-rate 1e-4"

IMAGE_VERSION=$(cat DOCKERIMAGEVERSION)
IMAGE_NAME="sckoo/bird-brain:v$IMAGE_VERSION"
OPTIONS=""
DEPENDENCIES=""
RUNARGS=""

for var in "$@"; do
    if [[ "$1" == "--gpu" ]]; then
        IMAGE_NAME="$IMAGE_NAME-gpu"
        OPTIONS="$OPTIONS --request-gpus 1"
        shift
    elif [[ "$1" == "--host" ]]; then
        # Using NLP cluster
        host="$2"
        OPTIONS="$OPTIONS --request-queue host=$host"
        shift
        shift
    elif [[ "$1" == "--resume" ]]; then
        uuid="$2"
        epoch="$3"
        DEPENDENCIES="$DEPENDENCIES old_model:$uuid/models _config.json:$uuid/config.json"
        RUNARGS="$RUNARGS --load-from-file old_model/saved_model_epoch-$epoch --config _config.json"
        shift
        shift
        shift
    fi
done


# Ensure we're on the right worksheet
# cl work main::bird-brain >/dev/null

# Re-upload src if it's newer than the last uploaded bundle named "src"
last_uploaded=$(date -r $(cl info -f created src) "+%Y%m%d%H%M")
touch -t $last_uploaded .last_uploaded
if [[ $(find src -newer .last_uploaded | wc -c) -ne 0 ]]; then
    echo "Changes found since src was last uploaded..."
    cl upload src
fi
rm .last_uploaded

command="cl run --tail -n run-train $OPTIONS --request-docker-image $IMAGE_NAME $DEPENDENCIES :src :newdata --- python src/run.py $RUNARGS ${@}"
echo $command
printf "New run bundle uuid: "
$command
