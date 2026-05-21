from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lrom_demo.core as core
import lrom_demo.rose_rom as rose_rom

print("core rose", core.rose.__file__)
print("central kinematics", core.central_kd_parameters()[0:4])
print("phase2 config", rose_rom.Phase2Config(n_train=4, n_phi=2, n_U=2))
