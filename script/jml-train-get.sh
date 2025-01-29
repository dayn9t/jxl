#!/usr/bin/env bash

if [[ x$1 != x ]]; then
    host=$1
else
    host=shtm1
fi

file=best.mod

echo 下载分类模型: $file

p=$(pwd)/$file
rsync -aP $host:"$p" "$p"

name=$(basename $(pwd)).pt
dt=$(date +"%Y-%m-%d")
dt_name="$dt"_"$name"

mv -f $file model_dir/"$dt_name"

cd model_dir || exit
rm "$name"
ln -s "$dt_name" "$name"
