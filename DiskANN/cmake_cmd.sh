cd ~/DiskANN
rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -mavx2 -mfma -fopenmp"

make -j$(nproc)
