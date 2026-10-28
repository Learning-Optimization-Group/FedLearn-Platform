"""Frozen generated projection study and order-independent public streams."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

SETTINGS = dict(dimension=512, train=4096, evaluation=8192, dataset_seed=20261027,
                clients=20, rounds=20, epochs=1, batch=128, lr=.03, clip=.05,
                epsilon=4., delta=1e-5)


def seed_for(seed, *parts):
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    raw = json.dumps([seed, *parts], separators=(',', ':'), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


@dataclass(frozen=True)
class Cell:
    block: str
    seed: int
    private: bool
    eta: float | None = None
    projection_seed: int | None = None
    steps: int = 0
    kind: str = ''
    label: str = ''

    def __post_init__(self):
        if self.family not in ('full', 'controlled', 'random', 'padded'):
            raise ValueError('unknown representation')
        if self.family == 'controlled' and self.eta not in (0., .25, .75, 1.):
            raise ValueError('invalid retained energy')
        if self.family in ('random', 'padded') and type(self.projection_seed) is not int:
            raise ValueError('projection seed required')
        if type(self.private) is not bool or self.steps not in (0,20,80,160):
            raise ValueError('invalid noise or attack budget')
        seed_for(self.seed)

    @property
    def family(self):
        return self.kind or self.block

    @property
    def width(self):
        return 512 if self.family in ('full', 'padded') else 128

    @property
    def name(self):
        return '_'.join(map(str, (self.block, self.family, self.eta, self.projection_seed,
                                  self.seed, 'gdp4' if self.private else 'none', self.steps, self.label)))

    def record(self):
        return dict(asdict(self), name=self.name, width=self.width)


def cells():
    out = []
    for seed in range(24001,24005):
        for private in (False,True):
            out.append(Cell('full',seed,private))
            out.extend(Cell('controlled',seed,private,eta=e) for e in (0.,.25,.75,1.))
            out.extend(Cell('random',seed,private,projection_seed=b) for b in range(20261101,20261106))
            out.append(Cell('padded',seed,private,projection_seed=20261101))
            for kind in ('full','random'):
                out.extend(Cell('attack',seed,private,kind=kind,steps=s,
                                projection_seed=20261101 if kind=='random' else None) for s in (20,80,160))
    return out


def qualification_cells():
    seed,b = 25001,20261111
    full = Cell('full',seed,False)
    return [full, replace(full,label='replay'), Cell('full',seed,True),
            *[Cell(kind,seed,p,projection_seed=b) for p in (False,True) for kind in ('random','padded')],
            *[Cell('attack',seed,False,kind='full',steps=s) for s in (20,80,160)],
            Cell('attack',seed,True,kind='random',projection_seed=b,steps=160),
            Cell('controlled',seed,True,eta=0.)]
