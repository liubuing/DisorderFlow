from openmm import Platform
import sys
print("OpenMM platforms:")
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    print(f"  {i}: {p.getName()} (speed={p.getSpeed()})")

try:
    p = Platform.getPlatformByName("OpenCL")
    print(f"OpenCL OK: {p.getSpeed()}")
except Exception as e:
    print(f"OpenCL FAIL: {e}")

try:
    p = Platform.getPlatformByName("CUDA")
    print(f"CUDA OK: {p.getSpeed()}")
except Exception as e:
    print(f"CUDA FAIL: {e}")

cpu = Platform.getPlatformByName("CPU")
print(f"CPU speed: {cpu.getSpeed()}")

print(f"CPU cores: {Platform.getNumProcessors()}")
