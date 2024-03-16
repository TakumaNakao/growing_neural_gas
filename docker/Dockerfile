FROM ubuntu:22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt update && apt upgrade -y &&\
    apt install -y \
    git \
    wget \
    unzip \
    python3-pip \ 
    cmake \
    libeigen3-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    x11-xserver-utils \
    gnuplot \
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    python3-dev \
    clang \
    libc++-dev \
    libc++abi-dev \
    libsdl2-dev \
    ninja-build \
    libxi-dev \
    libtbb-dev \
    libosmesa6-dev \
    libudev-dev \
    autoconf \
    libtool \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/alandefreitas/matplotplusplus \
    && cd matplotplusplus \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" -DMATPLOTPP_BUILD_EXAMPLES=OFF -DMATPLOTPP_BUILD_TESTS=OFF \
    && cmake --build . --parallel 2 --config Release \
    && cmake --install .

RUN git clone https://github.com/isl-org/Open3D \
    && cd Open3D \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install