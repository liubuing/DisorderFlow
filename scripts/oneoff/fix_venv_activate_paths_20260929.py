"""One-off: fix venv activate scripts still pointing at the old C: location.

The venv was created when the project lived at C:\\biological and moved to D:
without recreation. Repoint every hardcoded C:\\biological / C:/biological
reference in the activate-family scripts to D:. Bytes-level replace covers
both ASCII and UTF-16LE variants.
"""

from pathlib import Path

scripts = Path(__file__).resolve().parents[2] / 'venv' / 'Scripts'
targets = ('activate', 'activate.fish', 'activate.bat', 'activate.ps1',
           'deactivate', 'deactivate.fish', 'deactivate.bat', 'deactivate.ps1')

for p in sorted(scripts.iterdir()):
    if p.name.lower() not in targets:
        continue
    data = p.read_bytes()
    fixed = (data.replace(b'C:\\biological', b'D:\\biological')
                 .replace(b'C:/biological', b'D:/biological')
                 .replace(b'C\x00:\\\x00b\x00i\x00o\x00', b'D\x00:\\\x00b\x00i\x00o\x00')
                 .replace(b'C\x00:\x00\\\x00b\x00i\x00o\x00', b'D\x00:\x00\\\x00b\x00i\x00o\x00')
                 .replace(b'C\x00:\x00b\x00', b'D\x00:\x00b\x00')
                 .replace(b'C:/biological'.replace(b'/', b'\x00/\x00'), b'D:/biological'.replace(b'/', b'\x00/\x00')))
    if fixed != data:
        p.write_bytes(fixed)
        print('fixed:', p.name)
    else:
        print('no change:', p.name)
