#!/bin/bash
# Stos DPU na KV260 z Ubuntu 24.04 (jądro 6.8.0-1015-xilinx) — 07.11.2026.
# Uruchamiać NA PŁYTCE jako ubuntu (sudo bez hasła). Kolejność jak zrobiona
# ręcznie 07.11; skrypt zebrany po fakcie — w całości od zera nie był
# puszczony, każdy krok osobno działał.
#
# Dlaczego tyle budowania (szczegóły: HISTORIA.md, „KV260: DPU działa"):
# - Ubuntu 24.04 ma XRT 2.18 z pakietu, ale BEZ sterownika zocl
#   (xrt-dkms z universe to 2.8 z 2020 — usunięty), a libxrt_core 2.18
#   ukrywa stary interfejs C (xclOpen, xclAllocBO, xclExecBuf...), na
#   którym stoi VART 3.5;
# - zocl 2.16 nie kompiluje się na jądrze 6.8 (vm_flags, DRM_UNLOCKED...),
#   zocl 2.18 tak;
# - VART 3.5 + gcc 13: brak <cstdint>, -Werror na nowych ostrzeżeniach.
set -e
XRT_TAG=202420.2.18.179
mkdir -p ~/src ~/build

# --- zależności
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git cmake \
  libgoogle-glog-dev libprotobuf-dev protobuf-compiler libjson-c-dev libboost-filesystem-dev \
  libboost-system-dev libboost-program-options-dev pybind11-dev python3-dev libeigen3-dev \
  libopencv-dev uuid-dev libdrm-dev libncurses-dev ocl-icd-opencl-dev opencl-headers \
  rapidjson-dev libelf-dev libssl-dev libudev-dev libyaml-dev libcurl4-openssl-dev \
  libsystemd-dev systemtap-sdt-dev libgtest-dev libffi-dev strace

# --- XRT 2.18 ze źródeł: zocl + libxrt_core/coreutil (wersja edge)
cd ~/src
[ -d XRT ] || git clone -q --depth 1 --branch $XRT_TAG --recurse-submodules --shallow-submodules \
  https://github.com/Xilinx/XRT.git
cd XRT/src/runtime_src
# xdp (profilowanie) wymaga emulatora sprzętu — niepotrzebne
sed -i "s/^\(\s*\)xrt_add_subdirectory(xdp)/\1# xrt_add_subdirectory(xdp)/" CMakeLists.txt
# stary interfejs C widoczny (2.18 oznacza go visibility("hidden"))
sed -i '52s/visibility("hidden")/visibility("default")/' core/include/deprecated/xrt.h
# budowa bez AIE (KV260 nie ma AIE) ma w 2.18 błąd: brak nagłówka i #ifdef
python3 - <<'EOF'
p = "core/edge/user/hwctx_object.cpp"; s = open(p).read()
if "aie_parser.h" not in s:
    s = s.replace('#include "core/edge/user/shim.h"\n',
                  '#include "core/edge/user/shim.h"\n#include "core/edge/common/aie_parser.h"\n', 1)
    a = "hwctx_object::reset_array() const\n{\n"
    i = s.index(a) + len(a); j = s.index("\n}\n", i)
    s = s[:i] + "#ifdef XRT_ENABLE_AIE\n" + s[i:j] + \
        "\n#else\n  throw xrt_core::error(std::errc::not_supported, __func__);\n#endif" + s[j:]
    open(p, "w").write(s)
EOF
# sterownik
cd core/edge/drm/zocl && make -j4
sudo mkdir -p /lib/modules/$(uname -r)/updates
sudo cp zocl.ko /lib/modules/$(uname -r)/updates/ && sudo depmod
echo zocl | sudo tee /etc/modules-load.d/zocl.conf
# biblioteki: XRT_NATIVE_BUILD=no to zmienna ŚRODOWISKA (bez niej powstaje wersja PCIe)
mkdir -p ~/build/xrt && cd ~/build/xrt
XRT_NATIVE_BUILD=no cmake ~/src/XRT/src -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-include cstdint" > cmake.log
make -j3 xrt_core xrt_coreutil > make.log
D=/opt/xilinx/xrt
sudo mkdir -p $D/lib $D/include $D/share/cmake
sudo cp -a runtime_src/core/edge/user/libxrt_core.so* runtime_src/core/common/libxrt_coreutil.so* $D/lib/
sudo cp -a /usr/include/xrt $D/include/
sudo cp ~/src/XRT/src/runtime_src/core/include/deprecated/xrt.h $D/include/xrt/deprecated/xrt.h
sudo cp -a /usr/share/cmake/XRT $D/share/cmake/
sudo sed -i "s#/usr/lib/libxrt_core.so.2.18.0#$D/lib/libxrt_core.so.2.18.0#; s#/usr/lib/libxrt_coreutil.so.2.18.0#$D/lib/libxrt_coreutil.so.2.18.0#" \
  $D/share/cmake/XRT/xrt-targets-none.cmake
echo $D/lib | sudo tee /etc/ld.so.conf.d/00-xrt-opt.conf && sudo ldconfig

# --- VART (Vitis AI 3.5): unilog, xir, target_factory, vart
cd ~/src
[ -d Vitis-AI ] || { git clone -q --depth 1 --branch v3.5 --filter=blob:none --sparse https://github.com/Xilinx/Vitis-AI.git
  (cd Vitis-AI && git sparse-checkout set src/vai_runtime/unilog src/vai_runtime/xir src/vai_runtime/target_factory src/vai_runtime/vart); }
cd Vitis-AI/src/vai_runtime
grep -rl "Werror" --include=CMakeLists.txt --include=*.cmake . | xargs -r sed -i "s/-Werror//g"
for p in unilog xir target_factory vart; do
  echo "=== $p $(date +%T)"
  rm -rf ~/build/$p && mkdir -p ~/build/$p && cd ~/build/$p
  # -I/opt/...: <xrt/xrt.h> inaczej bierze nagłówek z /usr/include (funkcje ukryte -> undefined reference)
  cmake ~/src/Vitis-AI/src/vai_runtime/$p -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TEST=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="-I/opt/xilinx/xrt/include -include cstdint -Wno-error" \
    -DBUILD_PYTHON=ON -DPYTHON_INSTALL_DIR=/usr/local/lib/python3.12/dist-packages \
    -DCMAKE_PREFIX_PATH=/usr/local -DXRT_DIR=/opt/xilinx/xrt/share/cmake/XRT > cmake.log 2>&1
  make -j3 > make.log 2>&1
  sudo make install > install.log 2>&1
  sudo ldconfig
  echo "ok $p"
done

# --- konfiguracja: xclbin bitstreamu z DPU (xmutil loadapp kv260-benchmark-b4096)
echo "firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin" | sudo tee /etc/vart.conf
sudo usermod -aG render ubuntu
echo "=== KONIEC $(date +%T)"
