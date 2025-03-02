#!/usr/bin/env bash

title=招牌有效性分类
name=valid
data=dataset
models="2c-yolov8m-cls"
exe=/home/jiang/py/jxl/script/y8-train.sh
$exe classify $title $name $models $data
