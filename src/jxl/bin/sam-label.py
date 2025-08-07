#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


def main(video_file:Path, label_folder:Path):
    capture = Capture(file)
    print(file)
    assert capture.is_opened()
    ok = capture.set_fps(0.2)
    assert ok
    assert capture.fps() == 0.2
    print("fps:", capture.fps())
    size = capture.video_size()
    print("size:", size)



if __name__ == "__main__":
    main()
