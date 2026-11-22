import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disasters import core
from disasters.renderer import W, H
import numpy as np
sc = core.Scene(W, H)
from disasters.tornado import Data
from disasters.renderer import Painter, Timeline
D=Data(); TL=Timeline(D); P=Painter(D,TL)
sc.set_texture(P.final_comp())
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
for name,(p,y,d,tx,tz) in {"a":(54,0,3.6,0,0.15),"b":(48,-12,3.8,0,0.2),"c":(60,10,3.5,0,0.08),"d":(82,0,4.0,0,0.0)}.items():
    sc.render(core.Cam(tx,0,tz,d,p,y)).save(out/f"{name}.jpg",quality=88)
