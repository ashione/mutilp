#!/bin/bash

if [$# != 3 ] 
then 
    echo 'error'
else
    echo 'generating images in $1 with $2 training or test split into file $3'
    ./genarateLabel.py $1 orginalList.txt
    ./conver_test_or_test.py videoNewList.txt orginalList.txt $2 $3
fi
