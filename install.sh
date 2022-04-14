#!/bin/bash

cd calvin_env/tacto
pip install -e .
cd ..
git apply ../rename_hulc.patch
pip install -e .
cd ..
pip install -e .
