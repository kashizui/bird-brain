#!/bin/bash

IMAGE_VERSION=$(cat DOCKERIMAGEVERSION)
IMAGE_NAME="sckoo/bird-brain:v$IMAGE_VERSION"

if [[ "$1" == "--gpu" ]]; then
    ARGS="--request-docker-image $IMAGE_NAME-gpu --request-gpus 1"
    shift
else
    ARGS="--request-docker-image $IMAGE_NAME"
fi

# Ensure we're on the right worksheet
cl work main::bird-brain >/dev/null

# Re-upload src if it's newer than the last uploaded bundle named "src"
last_uploaded=$(date -r $(cl info -f created src) "+%Y%m%d%H%M")
touch -t $last_uploaded .last_uploaded
if [[ $(find src -newer .last_uploaded | wc -c) -ne 0 ]]; then
    echo "Changes found since src was last uploaded..."
    cl upload src
fi
rm .last_uploaded

command="cl run --tail $ARGS :src :data --- python src/basic_model.py ${@}"
echo $command
printf "New run bundle uuid: "
$command
