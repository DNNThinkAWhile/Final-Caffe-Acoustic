#!/bin/bash
IFS=$'\n'
echo "Start!"
for p in $(cat example);
do
    echo $p;
done
