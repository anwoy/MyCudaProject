rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)  # The -j flag uses all your CPU cores to compile faster!
