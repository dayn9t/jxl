#!/usr/bin/env bash

title=店招检测
name=sigh
data=dataset.yaml
#models="yolov3-sppu yolov5m6u yolov5mu yolov8n yolov8l yolov8m"
models="yolov8m"

exe=/home/jiang/py/jxl/script/y8-train.sh
$exe detect $title $name $models $data
