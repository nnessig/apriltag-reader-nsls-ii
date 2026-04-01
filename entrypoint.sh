#!/usr/bin/env bash
set -e

source /opt/ros/${ROS_DISTRO}/setup.bash

# If Vimba X is mounted and vmbpy is not yet installed in the container,
# try to install the wheel automatically.
if ! python3 -c "import vmbpy" >/dev/null 2>&1; then
    if compgen -G "/opt/VimbaX/api/python/vmbpy-*.whl" > /dev/null; then
        echo "[INFO] Found Vimba X Python wheel. Installing vmbpy..."
        python3 -m pip install /opt/VimbaX/api/python/vmbpy-*.whl
    fi
fi

exec "$@"