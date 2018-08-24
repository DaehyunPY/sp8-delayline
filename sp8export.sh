#!/bin/bash
config=$(realpath "$1")
wdir=$(dirname "$config")
docker run \
    --rm \
    --interactive \
    --publish-all \
    --volume "$wdir":"$wdir" \
    daehyunpy/sp8-delayline python -m sp8export "$config" | tee "$1.log"
