#!/bin/bash
# VOT
git clone https://github.com/jvlmdr/trackdat.git
cd trackdat
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
bash scripts/unpack_vot.sh dl/vot2018 ../VOT2018
cp dl/vot2018/list.txt ../VOT2018/
cd .. && rm -rf ./trackdat

# OTB2015
mkdir OTB2015 && cd OTB2015
baseurl="http://cvlab.hanyang.ac.kr/tracker_benchmark"
wget "$baseurl/datasets.html"
cat datasets.html | grep '\.zip' | sed -e 's/\.zip".*/.zip/' | sed -e s'/.*"//' >files.txt
cat files.txt | xargs -n 1 -P 8 -I {} wget -c "$baseurl/{}"
ls *.zip | xargs -n 1 unzip
rm -r __MACOSX/
cd ..
