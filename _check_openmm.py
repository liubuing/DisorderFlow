import openmm
import numpy
print(f"numpy {numpy.__version__}")
print(f"openmm {openmm.__version__}")
from openmm import Platform
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    print(f"  {i}: {p.getName()} speed={p.getSpeed()}")
print("OK")
