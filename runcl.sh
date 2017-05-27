#!/bin/bash

# Ensure we're on the right worksheet
cl work main::bird-brain >/dev/null

# Re-upload src if it's newer than 
last_uploaded=$(date -r $(cl info -f created src) "+%Y%m%d%H%M")
touch -t $last_uploaded .last_uploaded
if [[ $(find src -newer .last_uploaded | wc -c) -ne 0 ]]; then
    echo "Changes found since src was last uploaded..."
    cl upload src
fi
rm .last_uploaded

command="cl run --tail --request-docker-image sckoo/bird-brain:v3 :src :data --- python src/basic_model.py ${@}"
echo $command
printf "New run bundle uuid: "
$command
