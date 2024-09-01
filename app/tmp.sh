#!/bin/bash
pip install cmake
git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE/
pip install .
cd ../
rm -rf Multicore-TSNE
