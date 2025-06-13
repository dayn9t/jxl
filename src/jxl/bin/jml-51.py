#!/opt/ias/env/bin/python

import fiftyone as fo
import fiftyone.zoo as foz


def load():
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    load()
