#!/usr/bin/env bash

function show_help() {
  echo -e "\nJML图像分类器训练程序，用法:"
  echo -e "    jml-train-ic.sh <名称> <类别数> <迭代次数> <批尺寸> <网络框架>\n"
}

if [[ x$5 == x ]]; then
  show_help
  exit 1
fi

name=$1
num_classes=$2
epochs=$3
batch_size=$4
arch=$5

echo "模型=$name 迭代=$epochs 批=$batch_size"

echo 始于: $(date +"%Y-%m-%d %H:%M:%S")

train=/opt/ias/env/lib/pyias/jml/bin/img-train.py
echo pwd: $(pwd)
echo cmd: $train -a "$arch" --num_classes "$num_classes" --pretrained -b "$batch_size" --epochs "$epochs" dataset
$train -a "$arch" --num_classes "$num_classes" --pretrained -b "$batch_size" --epochs "$epochs" dataset
mv best.pth best.mod

#mkdir archs
#cp best.mod archs/"$arch".mod

echo 止于: $(date +"%Y-%m-%d %H:%M:%S")
