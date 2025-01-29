#!/usr/bin/env bash

file=best.mod

name=$(basename $(pwd))
dt=$(date +"%Y-%m-%d")
dt_name="$dt"_"$name"

mv -f $file model_dir/"$dt_name"

cd model_dir || exit
rm "$name"
ln -s "$dt_name" "$name"
