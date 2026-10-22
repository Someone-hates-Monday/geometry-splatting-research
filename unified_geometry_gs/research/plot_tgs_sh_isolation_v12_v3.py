"""Layout-only successor; keep both prior plots and all original data untouched."""
from pathlib import Path

source = Path(__file__).with_name('plot_tgs_sh_isolation_v12_v2.py').read_text()
source = source.replace('source_effects_v2', 'source_effects_v3')
source = source.replace('figure_metadata_v2', 'figure_metadata_v3')
source = source.replace("b.set_xlim(-.0057,.002);", "b.set_xlim(-.0057,.002);b.set_xticks([-.004,-.002,0,.002]);")
# The plot's provenance must identify this actual driver, not the previous version.
exec(compile(source, str(Path(__file__)), 'exec'))
