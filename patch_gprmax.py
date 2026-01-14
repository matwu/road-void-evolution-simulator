#!/usr/bin/env python3
"""gprMax utilities.py patch script"""
import re
from pathlib import Path
import site

# Find utilities.py
utilities_path = None
for site_path in site.getsitepackages():
    matches = list(Path(site_path).glob('gprMax*/gprMax/utilities.py'))
    if matches:
        utilities_path = matches[0]
        break

if not utilities_path:
    print('utilities.py not found')
    exit(1)

print(f'Patching {utilities_path}...')

with open(utilities_path, 'r') as f:
    content = f.read()

# Replace get_host_info function with safer version
safer_function = '''def get_host_info():
    """Get information about the host machine.

    Returns:
        hostinfo (dict): Host information.
    """
    import platform
    import psutil

    hostinfo = {}
    hostinfo['hostname'] = platform.node()
    hostinfo['machineID'] = platform.machine()
    hostinfo['sockets'] = 1
    hostinfo['cpuID'] = platform.processor() or platform.machine()
    hostinfo['physicalcores'] = psutil.cpu_count(logical=False) or 1
    hostinfo['logicalcores'] = psutil.cpu_count(logical=True) or 1
    hostinfo['hyperthreading'] = hostinfo['logicalcores'] > hostinfo['physicalcores']
    hostinfo['osversion'] = platform.platform()
    hostinfo['ram'] = psutil.virtual_memory().total

    return hostinfo'''

# Replace entire function
pattern = r'def get_host_info\(\):.*?return hostinfo'
content_new = re.sub(pattern, safer_function, content, flags=re.DOTALL)

if content_new != content:
    with open(utilities_path, 'w') as f:
        f.write(content_new)
    print('Patch applied successfully!')
else:
    print('Pattern not found or already patched')
