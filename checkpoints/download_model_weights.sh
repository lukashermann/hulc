#!/bin/bash
# Download, Unzip, and Remove zip
if [ "$1" = "D" ]
then

    echo "Downloading HULC Checkpoint for task_D_D ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/HULC_D_D.zip
    unzip HULC_D_D.zip && rm HULC_D_D.zip
    echo "finished!"
elif [ "$1" = "ABC" ]
then

    echo "Downloading HULC Checkpoint for task_ABC_D ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/HULC_ABC_D.zip
    unzip HULC_ABC_D.zip && rm HULC_ABC_D.zip
    echo "finished!"

elif [ "$1" = "ABCD" ]
then

    echo "Downloading HULC Checkpoint for task_ABCD_D ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/HULC_ABCD_D.zip
    unzip HULC_ABCD_D.zip && rm HULC_ABCD_D.zip
    echo "finished!"

else
    echo "Failed: Usage download_model_weights.sh D | ABC | ABCD"
    exit 1
fi