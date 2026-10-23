# BioEmu2 examples

This directory contains executable examples as VS Code Python Interactive files
(`.py` files with `# %%` cell markers). See the
[notebook quickstart](../README.md#notebook-quickstart) to run them.

## Example system YAML files

Example system YAML files are in
[`src/bioemu2/assets/examples/systems/`](../src/bioemu2/assets/examples/systems/).
Pass one to a CLI command with `--system`:

```bash
bioemu2 sample-equilibrium \
  --system src/bioemu2/assets/examples/systems/chignolin.yaml \
  --out outputs/chignolin
```

| File | What it demonstrates |
| --- | --- |
| [`chignolin.yaml`](../src/bioemu2/assets/examples/systems/chignolin.yaml) | A 10-residue protein (chignolin variant CLN025): the smallest system, useful for quick tests. |
| [`ake.yaml`](../src/bioemu2/assets/examples/systems/ake.yaml) | A single 214-residue protein chain defined by its sequence: *E. coli* adenylate kinase (PDB 4AKE). |
| [`meteorin_like_7tlw.yaml`](../src/bioemu2/assets/examples/systems/meteorin_like_7tlw.yaml) | A protein with two disulfide bonds declared under `bonds` (PDB 7TLW, chain A). |
| [`myoglobin_101m.yaml`](../src/bioemu2/assets/examples/systems/myoglobin_101m.yaml) | A protein with three ligands identified by CCD code (sulfate, heme, and NBN), and a contact that keeps the heme iron atom within 6 Å of residue 96 (PDB 101M). |
| [`shp2_3tkz.yaml`](../src/bioemu2/assets/examples/systems/shp2_3tkz.yaml) | A protein bound to two copies of a peptide: one entity with two chain IDs, a mapping to author chain IDs, and a contact between one peptide and the binding pocket (PDB 3TKZ). |

In an installed package, the same files are under
`bioemu2.paths.EXAMPLES_DIR / "systems"`. See
[define molecular systems](../docs/system.md) for the full system format,
including [contact conditioning](../docs/system.md#define-contact-conditioning).
