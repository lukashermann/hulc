#!/bin/bash
# Download, Unzip, and Remove zip
if [ "$1" = "D" ]
then

    echo "Downloading Language Embeddings for task_D_D ..."
    cd task_D_D
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/D_D_lang_embs_train.zip
    unzip D_D_lang_embs_train.zip && rm D_D_lang_embs_train.zip
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/D_D_lang_embs_val.zip
    unzip D_D_lang_embs_val.zip && rm D_D_lang_embs_val.zip
    echo "finished!"
elif [ "$1" = "ABC" ]
then

    echo "Downloading Language Embeddings for task_ABC_D ..."
    cd task_ABC_D
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/ABC_D_lang_embs_train.zip
    unzip ABC_D_lang_embs_train.zip && rm ABC_D_lang_embs_train.zip
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/ABC_D_lang_embs_val.zip
    unzip ABC_D_lang_embs_val.zip && rm ABC_D_lang_embs_val.zip
    echo "finished!"

elif [ "$1" = "ABCD" ]
then

    echo "Downloading Language Embeddings for task_ABCD_D ..."
    cd task_ABCD_D
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/ABCD_D_lang_embs_train.zip
    unzip ABCD_D_lang_embs_train.zip && rm ABCD_D_lang_embs_train.zip
    wget http://hulc.cs.uni-freiburg.de/language_embeddings/ABCD_D_lang_embs_val.zip
    unzip ABCD_D_lang_embs_val.zip && rm ABCD_D_lang_embs_val.zip
    echo "finished!"

else
    echo "Failed: Usage download_lang_embeddings.sh D | ABC | ABCD"
    exit 1
fi