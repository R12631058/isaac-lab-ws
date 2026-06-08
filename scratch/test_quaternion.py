from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

import sys
from pxr import Gf

with open("scratch/quaternion_results.txt", "w") as f:
    f.write("Testing Gf types...\n")
    try:
        q1 = Gf.Quaternion(1, Gf.Vec3d(0))
        q2 = Gf.Quaternion(1, Gf.Vec3d(0))
        f.write(f"Quaternion multiplication: {q1 * q2}\n")
    except Exception as e:
        f.write(f"Quaternion multiplication failed: {e}\n")

    try:
        qd1 = Gf.Quatd(1, 0, 0, 0)
        qd2 = Gf.Quatd(1, 0, 0, 0)
        f.write(f"Quatd multiplication: {qd1 * qd2}\n")
    except Exception as e:
        f.write(f"Quatd multiplication failed: {e}\n")

    try:
        qf1 = Gf.Quatf(1, 0, 0, 0)
        qf2 = Gf.Quatf(1, 0, 0, 0)
        f.write(f"Quatf multiplication: {qf1 * qf2}\n")
    except Exception as e:
        f.write(f"Quatf multiplication failed: {e}\n")

    f.write(f"Quatd dir: {dir(Gf.Quatd)}\n")
    f.write(f"Quaternion dir: {dir(Gf.Quaternion)}\n")

app.close()
