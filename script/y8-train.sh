#!/usr/bin/env bash

function show_help() {
    echo -e "\nYOLOv8训练程序，用法:"
    echo -e "    y8-train.sh <TASK> <TITLE> <NAME> <MODELS> <DATA>\n"
}

if [[ x$5 == x ]]; then
    show_help
    exit 1
fi

task=$1
title=$2
name=$3
models=$4
models=${models//,/ }
data=$(pwd)/$5

epochs=400
batch=-1
pretrained=True
device=0
#resume=1

project=training
dst_mod=best.mod

chmod 777 nohup.out

echo "$title" 模型训练 @ $(pwd)
#source /opt/ias/env/bin/activate

rm training -rf
rm $dst_mod

for model in $models; do
    echo yolo "$task" train data="$data" model=model_cfg/"$model".yaml epochs=$epochs batch=$batch \
        device=$device project=$project name="$model" pretrained=$pretrained
    yolo "$task" train data="$data" model=model_cfg/"$model".yaml epochs=$epochs batch=$batch \
        device=$device project=$project name="$model" pretrained=$pretrained
    echo $(date +"%Y-%m-%d %H:%M:%S") "$name $model => $dst_mod DONE"
    mv training/"$model"/weights/best.pt $dst_mod
    sleep 20
done
