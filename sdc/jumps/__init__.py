"""3D jump-course sim (MuJoCo)."""

import os

# Off-screen 3D rendering (render3d.py) uses software OpenGL: needs `apt-get install -y libosmesa6`.
# Must be set before mujoco is first imported.
os.environ.setdefault("MUJOCO_GL", "osmesa")
