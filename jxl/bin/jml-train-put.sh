#!/usr/bin/env bash

host=shtm1
p=$(pwd)/

echo "推送目录: $p => $host"

ssh $host "mkdir -p $p"
rsync -av --delete "$p" $host:"$p"
