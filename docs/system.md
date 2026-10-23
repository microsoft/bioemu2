# Define molecular systems

<!-- BEGIN GENERATED TOC -->
## Contents

- [Create a protein system](#create-a-protein-system)
- [Represent unidentified and modified residues](#represent-unidentified-and-modified-residues)
- [Describe a branched molecule](#describe-a-branched-molecule)
- [Add ligands](#add-ligands)
- [Load a deposited structure definition](#load-a-deposited-structure-definition)
  - [Metadata used when importing mmCIF](#metadata-used-when-importing-mmcif)
- [Read and write YAML](#read-and-write-yaml)
- [Add a non-standard covalent bond](#add-a-non-standard-covalent-bond)
- [Define contact conditioning](#define-contact-conditioning)
- [Inspect chains and topology](#inspect-chains-and-topology)
<!-- END GENERATED TOC -->

`System` describes the molecular composition passed to BioEmu2. It contains:

- one or more molecular entities;
- optional non-standard covalent bonds; and
- an optional mapping from canonical chain IDs to author chain IDs.

To start from a working file, browse the
[example system YAML files](../examples/README.md#example-system-yaml-files).
They cover a small protein, ligands, disulfide bonds, multiple chains, and
contacts.

See [`examples/sample_structures.py`](../examples/sample_structures.py) for an
executable workflow that loads a `System` and uses it for structure prediction
and equilibrium sampling.

If you already have a Boltz input file, [convert Boltz input YAML](boltz.md)
to this system format with `bioemu2 convert-boltz INPUT.yaml --out
OUTPUT.yaml`. Review the conversion warnings for fields that cannot be
represented.

## Create a protein system

Create a `Protein` from its one-letter amino-acid sequence and assign each copy
of that protein a chain ID with `asym_ids`:

```python
from bioemu2 import Protein, System

trp_cage = System(
    [
        Protein(
            sequence="NLYIQWLKDGGPSSGRPPPS",  # Trp-cage (PDB 1L2Y)
            asym_ids=["A"],
        )
    ]
)
```

The first argument to `System` corresponds to the `entities` field in the equivalent YAML:

```yaml
entities:
  - entity_type: protein
    sequence: NLYIQWLKDGGPSSGRPPPS
    asym_ids: [A]
```

`System` validates its contents when it is created. The `asym_ids` values used
as canonical chain IDs must be unique across the system and no more than four
characters long.

The same entity definition can describe several identical chains:

```python
homodimer = System(
    [
        Protein(
            sequence="RMKQLEDKVEELLSKNYHLENEVARLKKLVGER",  # GCN4 leucine zipper (PDB 2ZTA)
            asym_ids=["A", "B"],
        )
    ]
)
```

Equivalent YAML:

```yaml
entities:
  - entity_type: protein
    sequence: RMKQLEDKVEELLSKNYHLENEVARLKKLVGER
    asym_ids: [A, B]
```

BioEmu2 also supports `RNA`, `DNA`, `Ligand`, and `Branched` entities.

For a system defined only by polymer sequences and ligands, use
`System.from_description()` to assign chain IDs in input order:

```python
from bioemu2 import System

protein_with_ligand = System.from_description(
    proteins=["NLYIQWLKDGGPSSGRPPPS"],  # Trp-cage (PDB 1L2Y)
    ligand_ccd_codes=["ATP"],
)
assert protein_with_ligand.chain_ids == ["A", "B"]
```

Pass `proteins`, `rna`, `dna`, `ligand_ccd_codes`, or `ligand_smiles` as lists.
Chains receive IDs in that order, starting at `A`. Identical sequences become
one entity with multiple chain IDs. Use the `System` constructor instead when
you need modified residues, explicit chain IDs, or non-standard bonds.

## Represent unidentified and modified residues

Use `X` in a protein sequence or `N` in an RNA/DNA sequence for an unidentified
residue. If no entry in the `mods` field identifies that position, the model
uses its generic unknown-residue representation there.

A chemically modified or noncanonical residue with known identity should be
specified with a 1-based `res_id` and exactly one chemical identifier: a CCD
code or an InChI. This identifier determines the chemical structure built at
that position, regardless of the sequence letter. For a modification with a
meaningful biological parent, preserve its canonical letter when useful. For
example, use `T` for threonine modified as TPO. Use `X` or `N` when the
original identity is not relevant or there is no meaningful canonical parent
letter. The sequence is also used for MSA searches, so this choice can change
the MSA query and resulting features.

For example, a threonine phosphorylated to TPO can be specified as:

```python
from bioemu2 import ModifiedResidue, Protein, System

phosphorylated_protein = System(
    [
        Protein(
            sequence="MATGQ",
            asym_ids=["A"],
            mods=[ModifiedResidue(res_id=3, ccd_code="TPO")],
        )
    ]
)
```

Equivalent YAML:

```yaml
entities:
  - entity_type: protein
    sequence: MATGQ
    asym_ids: [A]
    mods:
      - res_id: 3
        ccd_code: TPO
```

See
[`examples/sample_a_phosphorylated_complex.py`](../examples/sample_a_phosphorylated_complex.py)
for an executable workflow that samples an SH2 domain bound to a
phosphotyrosine peptide and checks the phosphate contacts.

## Describe a branched molecule

Represent a branched entity, such as a glycan, with an ordered `mon_seq` of CCD
monomers and explicit bonds between its monomers and any attached entities.
Residue indices in bonds are 1-based positions in the relevant sequence or
`mon_seq`. For example, this YAML describes two NAG monomers linked to each
other and to an asparagine in a protein:

```yaml
entities:
  - entity_type: protein
    sequence: ACNE
    asym_ids: [A]
  - entity_type: branched
    asym_ids: [B]
    mon_seq:
      - ccd_code: NAG
      - ccd_code: NAG
bonds:
  - atom1: {name: ND2, res_name: ASN, res_id: 3, asym_id: A}
    atom2: {name: C1, res_name: NAG, res_id: 1, asym_id: B}
  - atom1: {name: O4, res_name: NAG, res_id: 1, asym_id: B}
    atom2: {name: C1, res_name: NAG, res_id: 2, asym_id: B}
```

Branched entities can be expressed and sampled, but are absent from current
model training data, so their generated structures are extrapolations.

See [`examples/sample_a_glycoprotein.py`](../examples/sample_a_glycoprotein.py)
for an executable workflow that builds an IgG1 Fc dimer with branched N-glycans
in Python, samples it, and compares the glycans with a crystal structure.

## Add ligands

A ligand is defined by exactly one of a Chemical Component Dictionary (CCD)
code, a SMILES string, or an InChI string:

```python
from bioemu2 import Ligand, Protein, System

protein_with_atp = System(
    [
        Protein(sequence="VIGPPGCGKGTQCEKLAEAGVEVVAA", asym_ids=["A"]),
        Ligand(ccd_code="ATP", asym_ids=["B"]),
        Ligand(ccd_code="MG", asym_ids=["C"]),
    ]
)
```

Equivalent YAML:

```yaml
entities:
  - entity_type: protein
    sequence: VIGPPGCGKGTQCEKLAEAGVEVVAA
    asym_ids: [A]
  - entity_type: ligand
    ccd_code: ATP
    asym_ids: [B]
  - entity_type: ligand
    ccd_code: MG
    asym_ids: [C]
```

Each represented heavy atom, including those in ligands and polymers, supplies
a formal-charge scalar to the atomistic embedding. CCD ligands (including
PDB/mmCIF imports) use CCD atom charges, SMILES ligands retain the charges
specified in the SMILES, and InChI ligands use RDKit's interpretation of the
supplied InChI (standard InChI can normalize mobile hydrogens). Templates are
cached separately by input type and chemical string.

Sampled `topology.cif` files record the model-input per-atom charges, including
zeros, in `_atom_site.pdbx_formal_charge` for retrospective analyses.

The model was trained with ligand charges from CCD-based PDB data and SDF-based
curated MD data. Even when inference uses the requested formal charges, it may
not faithfully sample conformations of that protonation or tautomer.

Chain IDs identify instances rather than chemical types. Identical molecules
can share one entity definition:

```python
two_adp_molecules = Ligand(ccd_code="ADP", asym_ids=["B", "C"])
```

The corresponding YAML entity is:

```yaml
- entity_type: ligand
  ccd_code: ADP
  asym_ids: [B, C]
```

## Load a deposited structure definition

Construct a system directly from an RCSB Protein Data Bank entry:

```python
from bioemu2 import System

myoglobin = System.from_pdb_id("101M")
```

By default, `from_pdb_id()` loads biological assembly 1 and includes ligands
other than water. Use `assembly_id=None` for the deposited asymmetric unit:

```python
asymmetric_unit = System.from_pdb_id(
    "1ATP",
    assembly_id=None,
    auth_asym_ids=["E"],
    include_ligands=False,
)
```

See the RCSB PDB-101 guide to
[biological assemblies](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies)
for the distinction between a deposited asymmetric unit and the
biologically relevant assembly generated from it.

Chains can be selected with either `auth_asym_ids` or `label_asym_ids`, but not
both. BioEmu2 uses label asym IDs as the system's canonical chain IDs and
records the corresponding author IDs in `system.asym_id_to_author_id`.

To exclude selected ligand types while keeping the others, pass their CCD
codes:

```python
system_without_sulfate = System.from_pdb_id(
    "1ATP",
    exclude_ligands=["SO4"],
)
```

The same conversion is available for a local, uncompressed mmCIF file:

```python
from pathlib import Path

system = System.from_cif(Path("structure.cif"))
```

A few entries denote an unidentified atom or ion with `UNX`, which is not
a chemically defined ligand. BioEmu2 represents it as a virtual site (`VS` in
the output CIF). To omit it, include `"UNX"` in `exclude_ligands`.

Noncanonical polymer monomers identified by a CCD code in the entry's sequence
records are included in `mods`. Ambiguous or conflicting residue identities
raise an error.

### Metadata used when importing mmCIF

`System.from_cif()` reads molecular composition from mmCIF metadata, not
from coordinates alone. A syntactically valid mmCIF file is not necessarily
sufficient for system import. The importer expects a single data block and
uses the following fields:

| Category | Fields | Use and requirement |
| --- | --- | --- |
| `_atom_site` | `label_entity_id`, `label_asym_id` | Required to associate entities with the label chain IDs present in the atom table. Entity IDs must be integers. |
| `_atom_site` | `auth_asym_id` | Required for the mapping from label to author chain IDs. Each label chain ID must map to one author chain ID. |
| `_entity_poly` | `entity_id`, `type`, `pdbx_seq_one_letter_code_can` | Required to import polymer entities: identifies their type and full canonical sequence, including unresolved residues. Supported types are `polypeptide(L)`, `polyribonucleotide`, and `polydeoxyribonucleotide`. |
| `_entity_poly` | `pdbx_seq_one_letter_code` | Unexpanded sequence establishes polymer positions and preserves identities of standard parenthesized CCD codes (including DNA). Nonstandard parenthesized codes are imported as modifications, even without monomer tables. When absent, a canonical sequence that disagrees with complete positional records cannot be aligned and raises an error. |
| `_pdbx_entity_nonpoly` | `entity_id`, `comp_id` | Required to identify non-polymer ligands by CCD code. Water (`HOH`) is omitted. |
| `_entity_poly_seq` | `entity_id`, `num`, `mon_id` | Optional entity-level monomer identities at 1-based polymer positions, used to identify modified monomers. |
| `_pdbx_poly_seq_scheme` | `asym_id`, `seq_id`, `mon_id` | Optional chain-level monomer identities at 1-based polymer positions; these take precedence over entity-level monomer identities. Standard overrides must agree with an aligned canonical residue even when the raw sequence contains a modified CCD code. Chains with differing residue identities are imported separately. |
| `_pdbx_struct_mod_residue` | `label_asym_id`, `label_seq_id`, `label_comp_id` | Optional explicit modification annotations at 1-based polymer positions. Conflicts with chain-level monomer identities or noncanonical entity-level identities raise an error. |

The fields listed for each optional category are read when that category is
present. Missing fields can cause system construction to fail. Entities in
the atom table without a supported polymer type or a non-polymer CCD identity are omitted
with a warning that identifies unsupported polymer types and absent entity
metadata categories; their identities are not inferred from atom or residue names.
A coordinate-only `_atom_site` table therefore does not provide the metadata
needed to reconstruct a system.

Branched polymers are not imported. `_struct_conn` records are ignored with
a warning: explicit inter-residue and inter-chain bonds are not imported, so
covalently bound ligands are not represented with their attachment bonds.
Crystallization additives and detergents identified as non-polymer ligands
are not automatically excluded.

## Read and write YAML

A system can be stored as a human-readable YAML file:

```yaml
entities:
  - entity_type: protein
    asym_ids: [A]
    sequence: NLYIQWLKDGGPSSGRPPPS  # Trp-cage (PDB 1L2Y)
    mods: []
bonds: []
```

Load and save the molecular composition with `System.from_yaml()` and
`system.to_yaml()`:

```python
from pathlib import Path

from bioemu2 import System

system = System.from_yaml(Path("system.yaml"))
system.to_yaml(Path("system-copy.yaml"))
```

The [example system YAML files](../examples/README.md#example-system-yaml-files),
also available under `bioemu2.paths.EXAMPLES_DIR / "systems"`, define example
molecular systems; some also specify [contacts](#define-contact-conditioning).
Use `System.from_yaml()` to load just the molecular composition.

## Add a non-standard covalent bond

Standard polymer bonds, such as peptide bonds, are inferred from the sequence.
Use `Bond` only for additional covalent connections:

```python
from bioemu2 import Atom, Bond, Protein, System

disulfide_bond = Bond(
    atom1=Atom(name="SG", res_name="CYS", res_id=3, asym_id="A"),
    atom2=Atom(name="SG", res_name="CYS", res_id=8, asym_id="A"),
)

crosslinked_protein = System(
    [
        Protein(sequence="ACCAAAACAA", asym_ids=["A"]),
    ],
    bonds=[disulfide_bond],
)
```

Equivalent YAML:

```yaml
entities:
  - entity_type: protein
    sequence: ACCAAAACAA
    asym_ids: [A]
bonds:
  - atom1: {name: SG, res_name: CYS, res_id: 3, asym_id: A}
    atom2: {name: SG, res_name: CYS, res_id: 8, asym_id: A}
```

Residue indices are 1-based. `System` checks that referenced chains and
residues exist and that polymer residue names match the sequence. Atom names
are resolved when the structure is constructed rather than validated when the
`System` is created.

Declare every disulfide: steered samplers restrain a declared SG–SG bond to its
reference length (see [`BondLengthPotential`](potentials.md#choose-from-the-available-potentials)) and exclude it
from the van der Waals potentials. With a steered sampler (`edm_gd`, `edm_fk`,
`edm_fkc`), the van der Waals potentials push undeclared disulfides apart; the
default `edm` sampler applies no potentials.

## Define contact conditioning

BioEmu2 represents a molecular system as a sequence of tokens: each conventional
polymer residue has one token, while each heavy atom of a ligand, branched
monomer, or modified residue has its own token. A contact identifies two tokens
that should be close during inference and sets a maximum distance for the pair.
It does not add a covalent bond and is not stored on `System`.

Specify each endpoint with a chain identifier and a 1-based residue index, or
with a chain identifier and an atom name for a ligand. Use named fields to make
these addresses clear:

```python
from bioemu2 import Contact, Protein, System
from bioemu2.data_interface import ContactToken

system = System(
    [
        Protein(sequence="NLYIQWLKDGGPSSGRPPPS", asym_ids=["A"]),  # Trp-cage (PDB 1L2Y)
    ]
)
contacts = [
    Contact(
        token1=ContactToken(asym_id="A", res_id=3),
        token2=ContactToken(asym_id="A", res_id=8),
        max_distance=6.0,
    )
]
system.validate_contacts(contacts)
```

To load both the system and its contacts, put them in the same YAML file:

```yaml
entities:
  - entity_type: protein
    sequence: NLYIQWLKDGGPSSGRPPPS
    asym_ids: [A]
contacts:
  - token1: {asym_id: A, res_id: 3}
    token2: {asym_id: A, res_id: 8}
    max_distance: 6.0
```

```python
from pathlib import Path

from bioemu2 import load_system_and_contacts

system, contacts = load_system_and_contacts(Path("conditioned-system.yaml"))
```

For a modified polymer residue, omit `atom_name` to select its `CA` (protein)
or `C1'` (RNA/DNA) token. To select an atomistic token in a modified polymer
residue or branched entity, provide `res_id` and `atom_name`.

To contact a protein residue with a ligand atom, include both chains in the
system and specify the ligand atom by name:

```python
from bioemu2 import Contact, EmuSession, Ligand, Protein, System, load_model
from bioemu2.data_interface import ContactToken

system = System(
    [
        Protein(sequence="NLYIQWLKDGGPSSGRPPPS", asym_ids=["A"]),  # Trp-cage (PDB 1L2Y)
        Ligand(smiles="CC", asym_ids=["L"]),
    ]
)
ligand_contact = Contact(
    token1=ContactToken(asym_id="A", res_id=3),
    token2=ContactToken(asym_id="L", atom_name="C1"),
)
emu = EmuSession(system, model=load_model(), mode="structure", contacts=[ligand_contact])
predictions = emu.sample(num_samples=8)
```

In a system YAML input, use the corresponding keys:

```yaml
entities:
  - entity_type: protein
    sequence: NLYIQWLKDGGPSSGRPPPS
    asym_ids: [A]
  - entity_type: ligand
    smiles: CC
    asym_ids: [L]
contacts:
  - token1: {asym_id: A, res_id: 3}
    token2: {asym_id: L, atom_name: C1}
```

The session validates the contacts against the system before embedding it.
For direct functional calls, see the [functional Python API](functional-api.md#sample-structures).

## Inspect chains and topology

`chain_ids` lists canonical chain IDs in model order:

```python
assert protein_with_atp.chain_ids == ["A", "B", "C"]
```

`topology` builds a coordinate-free `mdtraj.Topology` from the same
`AllAtomStructure` representation used for inference:

```python
topology = protein_with_atp.topology
print(topology.n_chains, topology.n_residues, topology.n_atoms)
```

`EmuSession.sample()` and `EmuSession.propagate()` return
`mdtraj.Trajectory` objects with their topology already attached. Use
`system.topology` when you need to inspect the system before sampling or
associate compatible coordinates with it explicitly.

As with the `res_id` values used for bonds and contacts, generated topologies
number residues from 1 within each chain. This includes ligands and branched
monomers in sampled `topology.cif` files. MDTraj's `residue.index` remains
zero-based across the whole topology; `save_frames()` preserves the numbering
of externally supplied trajectories when every chain has a nonempty ID.
Otherwise, its CIF writer renumbers residues from 1.
Loading a charged CIF through MDTraj can drop these formal charges; retain the
original `topology.cif` or `System` if you need to inspect the model-input
charges later.

For low-level inference code that needs an `AllAtomStructure`, first create a
`System` (directly, with `System.model_validate()` for a dictionary, or with
`System.from_yaml()`, `System.from_cif()`, or `System.from_pdb_id()`), then convert it:

```python
from bioemu2.structure import AllAtomStructure

structure = AllAtomStructure.from_system(protein_with_atp)
```
