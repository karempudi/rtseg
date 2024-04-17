
from rtseg.cellseg.utils.tiling.padding import PaddingTiler
from rtseg.cellseg.utils.tiling.resize import ResizeTiler
from rtseg.cellseg.utils.tiling.dynamic_overlap import DynamicOverlapTiler

tilers = {
    "dynamic_overlap": DynamicOverlapTiler,
    "padding": PaddingTiler,
    "resize": ResizeTiler
}

def get_tiler(tiler_type):
    assert tiler_type in tilers, "Tiler type must be one of \
            `dynamic_overlap`, `padding`, or `resize`."

    return tilers[tiler_type]