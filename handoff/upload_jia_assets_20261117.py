"""Upload only the newly completed Jia run to a separate Release."""
from pathlib import Path
import upload_research_assets as transfer

if __name__=='__main__':
    transfer.TAG='jia-trust-rebuild-2026-11-17'
    transfer.OUT=transfer.ROOT/'.asset-transfer/jia-20261117'
    transfer.OUT.mkdir(parents=True,exist_ok=True)
    transfer.GROUPS=['unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116']
    transfer.RELEASE_TITLE='Jia frozen-model paired experiment: completed, gates failed (2026-11-17)'
    transfer.RELEASE_NOTES='Separate asset package for the completed Jia paired experiment. All three gates failed. Contains frozen models, control/trust meshes, raw results, source snapshots and seals. Original 90-volume research-assets-2026-11-15 Release is unchanged. Completion requires ASSET_MANIFEST.json. See handoff/JIA_TRUST_REBUILD_CLOSEOUT_20261117.md.'
    transfer.main()
