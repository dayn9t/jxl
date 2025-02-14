#!/usr/bin/env bash

dir=jxl/bin
scrips="jxl_label.py jxl_label_clean.py jxl_split.py"
for script in $scrips; do
    poetry run pyinstaller --onefile --strip $dir/"$script"
done
