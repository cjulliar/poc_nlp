.s#!/bin/sh
docker build -t cyril4000/qsgc_gradio ./firstApp/.
docker push cyril4000/qsgc_gradio

az container create --resource-group RG_JULLIARDC --file deploy.yaml