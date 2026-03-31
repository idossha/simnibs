# Standalone SimNIBS 4.6 image with GUI support.
#
# docker build --platform linux/amd64 \
#   -f Dockerfile \
#   -t simnibs/simnibs:X.X.X .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System utilities + graphics/X11/Qt/XCB libraries for GUI
RUN apt-get update && apt-get install -y \
  wget \
  tar \
  bzip2 \
  libglib2.0-0 \
  # OpenGL / Mesa (software rendering via llvmpipe)
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  libglu1-mesa \
  libglapi-mesa \
  libosmesa6 \
  libegl1-mesa \
  libgles2-mesa \
  libgbm1 \
  mesa-utils \
  # X11
  libxt6 \
  libxext6 \
  libxrender1 \
  libxrandr2 \
  libxfixes3 \
  libxcursor1 \
  libxcomposite1 \
  libxdamage1 \
  libxi6 \
  libxmu6 \
  libxft2 \
  libice6 \
  libsm6 \
  x11-utils \
  # XCB (required by Qt5 xcb platform plugin)
  libxcb-xinerama0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-shape0 \
  libxkbcommon-x11-0 \
  # Qt5 (SimNIBS GUI)
  libqt5widgets5 \
  libqt5gui5 \
  libqt5core5a \
  libqt5svg5 \
  libqt5opengl5 \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Avoid OpenMP affinity crash during postinstall
ENV OMP_NUM_THREADS=1
ENV KMP_AFFINITY=disabled

# Install SimNIBS
RUN mkdir -p /simnibs \
  && wget https://github.com/simnibs/simnibs/releases/download/v4.6.0/simnibs_installer_linux.tar.gz -P /simnibs \
  && tar -xzf /simnibs/simnibs_installer_linux.tar.gz -C /simnibs \
  && /simnibs/simnibs_installer/install -s \
  && rm -rf /simnibs

ENV PATH="/root/SimNIBS-4.6/bin:$PATH"
ENV SIMNIBSDIR="/root/SimNIBS-4.6"

CMD ["bash"]
