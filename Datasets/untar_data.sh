#!/bin/bash

for i in {63..75}; do
  nohup tar -xf /mnt/spaceai-internal/panda-intervid/disk1/internvid/$i.tar -C /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Data/internvid/61-75 > logs/untar-$i.log 2>&1 &
done