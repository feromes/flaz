class FViz:
    def __init__(self, geom=None, meta=None, binfile=None):
        self.geom = geom
        self.meta = meta or {}
        self.binfile = binfile

    def save(self, outdir="."):
        # salva .parquet + .bin + .fviz.json + .meta.json
        pass
