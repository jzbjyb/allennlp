#!/usr/bin/env bash

inf=$1
out=$2
n=$3

if [ -f ${out} ]; then
  echo warning: ${out} exists
fi

for ((i=0;i<${n};i++))
do
  cat ${inf} >> ${out}
done
