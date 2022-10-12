#!/bin/bash

container_name="determined-env"

docker build -t $container_name .
