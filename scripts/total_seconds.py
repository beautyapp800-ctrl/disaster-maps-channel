"""Print the exact expected video length in seconds (whole frames at 24 fps)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
if os.environ.get("ENGINE") == "hurricane":
    from disasters.hurricane import FPS, Data, Timeline
elif os.environ.get("ENGINE") == "earthquake":
    from disasters.quake import FPS, Data, Timeline
else:
    from disasters.tornado import FPS, Data, Timeline

print(int(Timeline(Data()).total * FPS) / FPS)
