"""Versioned reader-only repair; preserve original failure and write a separate output tree."""
import hashlib
from pathlib import Path

original = Path(__file__).with_name('probe_tgs_sh_real_fields_v13.py')
source = original.read_text()
# Both source bytes and this driver are later frozen in the execution plan/manifest.
assert source.count(".numpy().astype(np.float64)") == 1
source = source.replace(".numpy().astype(np.float64)", ".detach().numpy().astype(np.float64)")
source = source.replace("experiments/tgs_sh_shell_v13/real_fields'", "experiments/tgs_sh_shell_v13/real_fields_v2'")
needle = "# IDs/coefficients are saved before this scene's field outputs exist."
source = source.replace(needle, """if model['scene']=='flowers':
            old=json.loads((OUT.parent/'real_fields/flowers_selection.json').read_text())
            assert ids.tolist()==old['ids']
        # IDs/coefficients are saved before this scene's field outputs exist.""")
source = source.replace("helper_sha256=sha(HELPER),models=files", "helper_sha256=sha(HELPER),original_script_sha256=sha(original),models=files")
source = source.replace("assert plan['script_sha256']==sha(Path(__file__))", "assert plan['original_script_sha256']==sha(original)\n    assert plan['script_sha256']==sha(Path(__file__))")
exec(compile(source, str(Path(__file__)), 'exec'))
