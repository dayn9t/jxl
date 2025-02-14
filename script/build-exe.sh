#!/usr/bin/env bash

poetry run pyinstaller --onefile --strip jxl/bin/jxl_label.py
poetry run pyinstaller --onefile --strip jxl/bin/jxl_label_clean.py
