from pathlib import Path

import mdtraj
import numpy as np
import pytest
import torch
from rdkit import Chem

from bioemu2.data_interface import DNA, RNA, ModifiedResidue, Protein, System
from bioemu2.data_interface.const import MolType, Token
from bioemu2.msa.colabfold import tokenize_msa
from bioemu2.physicality import compute_physicality_stats
from bioemu2.structure import AllAtomStructure, decode_atom_name, encode_atom_name
from bioemu2.structure.inference_inputs import (
    _create_all_edges,
    _create_bond_features,
    _inference_input_structure_from_nucleotide_sequence,
    _inference_input_structure_from_polymer_sequence_with_mods,
    _remove_capping_atoms,
    extract_rdkit_mol_from_structure,
    inference_input_structure_from_ccd,
    inference_input_structure_from_dna_sequence,
    inference_input_structure_from_inchi,
    inference_input_structure_from_protein_sequence,
    inference_input_structure_from_rna_sequence,
    topology_from_structure,
)
from bioemu2.structure.polymer_builder import PolymerStructureBuilder
from bioemu2.structure.templates import load_polymer_templates

ACETIC_ACID_SMILES = "CC(=O)O"


def check_if_bonds_equal(bond_index_a: torch.Tensor, bond_index_b: torch.Tensor) -> None:
    """
    Auxiliary function for checking if bond indices are equal. The ordering can follow different
    conventions and some routines return symmetric bonds ([i,j] and [j,i]) while others return
    unique ([i,j]). Because of this, we sort the bond pair tuples and use sets for checking.
    """

    def _bond_index_to_set(x: torch.Tensor) -> set[tuple[int, int]]:
        return set(tuple(sorted([i, j])) for i, j in x.T.tolist())

    assert _bond_index_to_set(bond_index_a) == _bond_index_to_set(bond_index_b)


def _assert_structure_match(
    structure_a: AllAtomStructure, structure_b: AllAtomStructure, check_bonds: bool = True
) -> None:
    """Check if the topology parts of two structure objects match."""
    assert structure_a.num_nodes == structure_b.num_nodes
    assert structure_a.num_residues == structure_b.num_residues
    assert torch.allclose(structure_a.element, structure_b.element)
    assert torch.equal(structure_a.residue_idx, structure_b.residue_idx)
    assert torch.equal(structure_a.residue_type, structure_b.residue_type)
    assert torch.equal(structure_a.name_encoding, structure_b.name_encoding)
    assert torch.equal(structure_a.atom_mask, structure_b.atom_mask)
    assert torch.equal(structure_a.asym_id, structure_b.asym_id)
    assert torch.equal(structure_a.entity_id, structure_b.entity_id)
    assert torch.equal(structure_a.sym_id, structure_b.sym_id)
    assert torch.equal(structure_a.mol_type, structure_b.mol_type)
    assert structure_a.sequence == structure_b.sequence


@pytest.mark.parametrize("num_nodes", [1, 10, 100])
def test_create_all_edges(num_nodes):
    """Check if automated edge creation is consistent."""
    edge_indices = _create_all_edges(num_nodes)

    assert edge_indices.shape == (2, num_nodes**2)

    target_edge_indices = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            target_edge_indices.append([i, j])
    target_edge_indices = torch.tensor(target_edge_indices).transpose(0, 1)

    assert torch.allclose(target_edge_indices, edge_indices)


@pytest.mark.parametrize("num_nodes", [2, 10, 100])
def test_create_bond_features(num_nodes):
    """Test creation of bond features based on bond indices."""
    bond_indices = torch.from_numpy(np.random.randint(0, num_nodes - 1, size=(2, num_nodes)))
    all_edges = _create_all_edges(num_nodes)
    bond_features = _create_bond_features(num_nodes, bond_indices)

    # Check if edges present in the bond list have bond features set to True.
    bond_list = bond_indices.transpose(0, 1).tolist()
    for idx in range(all_edges.shape[1]):
        edge = all_edges[:, idx]
        is_bonded = edge.tolist() in bond_list
        assert bond_features[idx].bool() == is_bonded


def test_inference_input_structure_from_protein_sequence(test_sequence, test_structure):
    """Check structure creation from a sequence using templates."""
    template_cache = load_polymer_templates()
    structure = inference_input_structure_from_protein_sequence(test_sequence, template_cache)
    _assert_structure_match(structure, test_structure)


def test_protein_topology_from_structure(test_structure, test_topology):
    """Test creation of topology from structure object."""
    topology = topology_from_structure(test_structure)
    assert topology == test_topology


def test_topology_from_structure_residue_renaming(test_structure, test_topology):
    """Test creation of topology with residue name overrides."""
    # Base topology.
    topology = topology_from_structure(test_structure)
    assert topology == test_topology

    # Residue label override.
    rename_residues: dict[int, str] = {1: "PUG", 3: "MUG"}

    # Topology with residue labels modified.
    topology_modded = topology_from_structure(test_structure, rename_residues=rename_residues)

    # Construct target residue labels
    target_residue_labels = [res.name for res in topology.residues]
    target_residue_labels = [
        rename_residues.get(idx, target_residue_labels[idx])
        for idx in range(len(target_residue_labels))
    ]

    current_residue_labels = [res.name for res in topology_modded.residues]
    assert target_residue_labels == current_residue_labels


def test_rna_topology_from_structure(test_structure_rna, test_topology_rna):
    """Test creation of topology from RNA structure object."""
    topology = topology_from_structure(test_structure_rna)
    assert topology == test_topology_rna


@pytest.mark.parametrize("polymer", [DNA, RNA])
@pytest.mark.parametrize("sequence", ["A", "ACNG"])
def test_nucleotide_topology_backbone_links(polymer, sequence):
    structure = AllAtomStructure.from_system(
        System(entities=[polymer(sequence=sequence, asym_ids="A")]), mode="structure"
    )
    names = structure.name_encoding.clone()
    first_op2 = torch.where((structure.residue_idx == 0) & structure.name_encoding_is_atom("OP2"))[
        0
    ][0]
    names[first_op2] = torch.tensor(encode_atom_name("OP3"))
    structure = structure.replace(name_encoding=names)
    topology = topology_from_structure(structure)
    bonds = {
        frozenset(((a.residue.index, a.name), (b.residue.index, b.name))) for a, b in topology.bonds
    }
    inter_residue = {pair for pair in bonds if len({residue for residue, _ in pair}) == 2}
    assert inter_residue == {
        frozenset(((index, "O3'"), (index + 1, "P"))) for index in range(len(sequence) - 1)
    }
    assert frozenset(((0, "P"), (0, "OP3"))) in bonds
    assert [atom.name for atom in topology.atoms] == [
        decode_atom_name(name) for name in structure.name_encoding
    ]


def test_nucleotide_links_preserve_chain_boundaries():
    structure = AllAtomStructure.from_system(
        System(
            entities=[
                DNA(sequence="AC", asym_ids="A"),
                RNA(sequence="GU", asym_ids="B"),
                Protein(sequence="AXA", asym_ids="C"),
            ]
        ),
        mode="structure",
    )
    topology = topology_from_structure(structure)
    links = [(a, b) for a, b in topology.bonds if a.residue.index != b.residue.index]
    assert all(a.residue.chain == b.residue.chain for a, b in links)
    na_links = {
        frozenset(((a.residue.index, a.name), (b.residue.index, b.name)))
        for a, b in links
        if a.residue.chain.index < 2
    }
    assert na_links == {
        frozenset(((0, "O3'"), (1, "P"))),
        frozenset(((2, "O3'"), (3, "P"))),
    }


def test_multichain_topology_from_structure(multichain_structure, multichain_topology):
    """Test creation of topology from multichain structure object."""
    topology = topology_from_structure(multichain_structure)
    assert topology == multichain_topology


def test_multichain_ligand_topology_from_structure(
    multichain_ligand_structure, multichain_ligand_topology
):
    """Test creation of topology from multichain structure object with ligand."""
    topology = topology_from_structure(multichain_ligand_structure)
    assert topology == multichain_ligand_topology


def test_multichain_ccd_modified_topology_from_structure(
    multichain_ccd_modified_structure, multichain_ccd_modified_topology
):
    """Test creation of topology from multichain structure object with CCD residue."""
    topology = topology_from_structure(multichain_ccd_modified_structure)
    assert topology == multichain_ccd_modified_topology


def test_extract_rdkit_mol_from_structure() -> None:
    smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # Caffeine
    structure = AllAtomStructure.from_system(System.from_description(ligand_smiles=[smiles]))
    extracted_mol = extract_rdkit_mol_from_structure(structure)
    assert Chem.MolToSmiles(extracted_mol) == smiles


def test_structure_has_charge() -> None:
    smiles = "[Na+].[Cl-]"  # Sodium chloride
    structure = AllAtomStructure.from_system(System.from_description(ligand_smiles=[smiles]))
    assert torch.equal(structure.charge, torch.tensor([1, -1], dtype=torch.int8))


@pytest.mark.parametrize(
    "smiles",
    [
        pytest.param("CC(=O)NCC(=O)OP(=O)([O-])O", id="acyl-phosphate"),
        pytest.param(
            "CC(=O)[N-]S(=O)(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
            id="acyl-sulfamate",
        ),
    ],
)
def test_smiles_ligand_retains_per_atom_formal_charge(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    structure = AllAtomStructure.from_system(System.from_description(ligand_smiles=[smiles]))

    assert structure.num_nodes == molecule.GetNumHeavyAtoms()
    assert structure.charge.tolist() == [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
    assert int(structure.charge.sum()) == -1


def test_inference_input_structure_from_nucleotide_sequence(test_sequence_rna, test_structure_rna):
    """Check structure creation from a RNA sequence using templates."""

    # Check if template is created correctly.
    template_cache = load_polymer_templates()
    structure_rna = _inference_input_structure_from_nucleotide_sequence(
        test_sequence_rna, template_cache, nucleotide_type=MolType.RNA
    )
    _assert_structure_match(structure_rna, test_structure_rna)

    structure_rna_direct = inference_input_structure_from_rna_sequence(
        test_sequence_rna, template_cache
    )
    _assert_structure_match(structure_rna_direct, test_structure_rna, check_bonds=False)

    # Check if DNA template creation on non-DNA nucleotides (U) fails.
    with pytest.raises(KeyError):
        _ = _inference_input_structure_from_nucleotide_sequence(
            "TU", template_cache, nucleotide_type=MolType.DNA
        )

    # Check if RNA template creation on non-DNA nucleotides (T) fails.
    with pytest.raises(KeyError):
        _ = _inference_input_structure_from_nucleotide_sequence(
            "TU", template_cache, nucleotide_type=MolType.RNA
        )

    # Check if DNA template from same sequence as DNA has the correct sugars
    # (should miss the O2' atoms).
    structure_dna = _inference_input_structure_from_nucleotide_sequence(
        test_sequence_rna, template_cache, nucleotide_type=MolType.DNA
    )
    structure_dna_direct = inference_input_structure_from_dna_sequence(
        test_sequence_rna, template_cache
    )
    _assert_structure_match(structure_dna_direct, structure_dna)

    # Each RNA monomer has exactly one O2' atom.
    assert structure_dna.num_nodes == structure_rna.num_nodes - len(test_sequence_rna)

    atom_names_rna = [decode_atom_name(ne) for ne in structure_rna.name_encoding]
    atom_names_dna = [decode_atom_name(ne) for ne in structure_dna.name_encoding]

    # Find difference in atom types.
    different_atom_type = set(atom_names_rna).difference(atom_names_dna)
    # There should only be one different atom type and it should be the ribose O2' which is there in
    # RNA but not in DNA.
    assert len(different_atom_type) == 1
    assert next(iter(different_atom_type)) == "O2'"


def test_inference_input_structure_from_ccd():
    """Check input structure generation from CCD."""
    ccd_code = "IUM"  # Uranyl ion.
    structure = inference_input_structure_from_ccd(ccd_code)

    target_element = torch.tensor([92, 8, 8], dtype=torch.long)
    target_name_encoding = torch.Tensor([encode_atom_name(an) for an in ["U", "O1", "O2"]]).long()
    target_charge = torch.Tensor([4, -1, -1]).to(torch.int8)
    target_token_center_atom_index = torch.arange(3).long()
    target_token_idx_at_atom = torch.arange(3).long()

    assert structure.num_nodes == 3
    assert structure.num_tokens == 3
    assert structure.num_residues == 1
    assert structure.coords.shape == (3, 3)
    assert torch.all(structure.mol_type == MolType.NONPOLYMER.value)
    assert torch.all(structure.asym_id == 0)
    assert torch.allclose(structure.element, target_element)
    assert torch.all(structure.residue_type == Token.UNK_PROT.value)
    assert torch.all(structure.residue_idx == 0)
    assert torch.allclose(structure.name_encoding, target_name_encoding)
    assert torch.allclose(structure.charge, target_charge)
    assert torch.all(structure.atom_mask)
    assert torch.allclose(structure.token_center_atom_index, target_token_center_atom_index)
    assert torch.allclose(structure.token_idx_at_atom, target_token_idx_at_atom)
    assert structure.template_coords.shape == (3, 3)


def test_inference_input_structure_from_inchi():
    """Check input structure generation from inchi."""
    inchi = "InChI=1S/U.2O/q+4;2*-1"  # Uranyl ion.
    structure = inference_input_structure_from_inchi(inchi)

    target_element = torch.tensor([92, 8, 8], dtype=torch.long)
    target_name_encoding = torch.Tensor([encode_atom_name(an) for an in ["U1", "O2", "O1"]]).long()
    target_charge = torch.Tensor([4, -1, -1]).to(torch.int8)
    target_token_center_atom_index = torch.arange(3).long()
    target_token_idx_at_atom = torch.arange(3).long()

    assert structure.num_nodes == 3
    assert structure.num_tokens == 3
    assert structure.num_residues == 1
    assert structure.coords.shape == (3, 3)
    assert torch.all(structure.mol_type == MolType.NONPOLYMER.value)
    assert torch.all(structure.asym_id == 0)
    assert torch.allclose(structure.element, target_element)
    assert torch.all(structure.residue_type == Token.UNK_PROT.value)
    assert torch.all(structure.residue_idx == 0)
    assert torch.allclose(structure.name_encoding, target_name_encoding)
    assert torch.allclose(structure.charge, target_charge)
    assert torch.all(structure.atom_mask)
    assert torch.allclose(structure.token_center_atom_index, target_token_center_atom_index)
    assert torch.allclose(structure.token_idx_at_atom, target_token_idx_at_atom)
    assert structure.template_coords.shape == (3, 3)


def test_structure_from_cif_system(pdb_files_dir: Path) -> None:
    """
    Check input structure generation from CIF file
    """
    system = System.from_cif(pdb_files_dir / "7WOI-assembly1.cif")
    structure = AllAtomStructure.from_system(system)

    assert len(torch.unique(structure.entity_id)) == 2  # Spa2 and EDO
    assert len(torch.unique(structure.asym_id)) == 3  # 1 copy of Spa2, 2 copies of EDO
    assert len(torch.unique(structure.mol_type)) == 2  # Spa2 (protein) and EDO (ligand)
    assert structure.coords.shape[1] == 3
    assert structure.template_coords is not None
    assert structure.template_coords.shape[1] == 3
    assert torch.all(structure.atom_mask)


@pytest.mark.parametrize(
    ["sequence", "modifications", "polymer_type"],
    [
        [
            "CGCGAANTTGCG",
            [
                (
                    7,
                    "inchi",
                    "InChI=1S/C9H13BrN3O7P/c10-4-2-13(9(15)12-8(4)11)7-1-5(14)6(20-7)3-19-21(16,17)18/h2,5-7,14H,1,3H2,(H2,11,12,15)(H2,16,17,18)/t5-,6+,7+/m0/s1",
                )
            ],
            "DNA",
        ]
    ],
)
def test_inference_input_polymer_sequence_with_mods(sequence, modifications, polymer_type):
    polymer_type_enum = MolType[polymer_type]
    st = _inference_input_structure_from_polymer_sequence_with_mods(
        sequence=sequence,
        modifications=modifications,
        template_cache=load_polymer_templates(),
        polymer_type=polymer_type_enum,
    )

    # Covalent ligands inherit some properties of the parent polymer
    assert (
        len(torch.unique(st.asym_id))
        == len(torch.unique(st.entity_id))
        == len(torch.unique(st.sym_id))
        == len(torch.unique(st.mol_type))
        == 1
    )
    assert st.mol_type[0].item() == polymer_type_enum.value

    # Check shapes are as expected
    assert (
        st.num_tokens
        == st.asym_id.shape[0]
        == st.entity_id.shape[0]
        == st.sym_id.shape[0]
        == st.distogram_index.shape[0]
        == st.token_center_atom_index.shape[0]
    )
    assert st.num_residues == len(sequence)

    assert (
        st.num_nodes
        == st.coords.shape[0]
        == st.element.shape[0]
        == st.charge.shape[0]
        == st.residue_type.shape[0]
        == st.name_encoding.shape[0]
        == st.atom_mask.shape[0]
        == st.template_coords.shape[0]
        == st.token_idx_at_atom.shape[0]
        == st.mol_type.shape[0]
        == st.residue_idx.shape[0]
    )

    # Test if residues are labeled correctly. DNA should use prefix `D` and an inchi modification
    # will get the `UNK` label.
    residue_names = ["UNK" if res == "N" else f"D{res}" for res in sequence]
    expected_residue_labels = [residue_names[ridx] for ridx in st.residue_idx.cpu().tolist()]
    assert expected_residue_labels == st.residue_labels


@pytest.mark.parametrize(
    ("sequence", "modifications", "expected_labels"),
    [
        ("N", [(1, "ccd", "ATP")], ["ATP"]),
        ("NN", [(1, "ccd", "ATP"), (2, "ccd", "ATP")], ["ATP", "ATP"]),
    ],
)
def test_entirely_modified_dna_polymer_builds(
    sequence: str, modifications: list[tuple[int, str, str]], expected_labels: list[str]
) -> None:
    structure = _inference_input_structure_from_polymer_sequence_with_mods(
        sequence=sequence,
        modifications=modifications,
        template_cache=load_polymer_templates(),
        polymer_type=MolType.DNA,
    )
    assert structure.num_residues == len(sequence)
    assert structure.num_nodes > 0
    assert torch.all(structure.mol_type == MolType.DNA.value)
    assert [
        structure.residue_labels[structure.residue_idx.tolist().index(i)]
        for i in range(len(sequence))
    ] == expected_labels
    assert topology_from_structure(structure).n_residues == len(sequence)


def test_empty_polymer_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="without residues and atoms"):
        PolymerStructureBuilder(MolType.DNA).build()
    with pytest.raises(ValueError, match="without residues and atoms"):
        _inference_input_structure_from_polymer_sequence_with_mods(
            sequence="", template_cache=load_polymer_templates(), polymer_type=MolType.RNA
        )


def test_remove_capping_atoms():
    """Check if capping atoms are removed as intended."""
    base_structure = inference_input_structure_from_ccd("TPO")
    base_atom_names = set(base_structure.atom_names)

    decapped_structure = _remove_capping_atoms(base_structure)
    decapped_atom_names = set(decapped_structure.atom_names)

    assert base_atom_names.difference(decapped_atom_names) == set(["OXT"])


def test_inference_input_structure_from_protein_sequence_with_ccd_mods():
    """Sanity check to see whether capping OXTs are removed when handling modified residues."""
    sequence = "AXA"
    modifications = [(2, "ccd", "NLE")]
    structure = inference_input_structure_from_protein_sequence(
        sequence, template_cache=load_polymer_templates(), modifications=modifications
    )

    assert "OXT" not in structure.atom_names

    # Also check expected residue labels.
    expected_resnames = ["ALA", "NLE", "ALA"]
    target_residue_labels = [
        expected_resnames[ridx] for ridx in structure.residue_idx.cpu().tolist()
    ]
    assert structure.residue_labels == target_residue_labels


def test_modified_protein_topology_checks_sidechain_and_peptide_bonds():
    system = System(
        entities=[
            Protein(
                sequence="AXA",
                asym_ids=["A"],
                mods=[ModifiedResidue(res_id=2, ccd_code="TPO")],
            )
        ]
    )
    structure = AllAtomStructure.from_system(system)
    topology = system.topology
    previous, modified, following = list(topology.residues)
    bonds = {frozenset((a.index, b.index)) for a, b in topology.bonds}

    assert {frozenset(pair) for pair in structure.bond_index.T.tolist()} <= bonds
    for first, second in ((previous, modified), (modified, following)):
        carbon = next(atom for atom in first.atoms if atom.name == "C")
        nitrogen = next(atom for atom in second.atoms if atom.name == "N")
        assert frozenset((carbon.index, nitrogen.index)) in bonds
    assert len(bonds) == topology.n_bonds

    trajectory = mdtraj.Trajectory(
        structure.template_coords.cpu().numpy()[None, :, :] / 10.0, topology
    )
    baseline = compute_physicality_stats(trajectory)
    phosphorus = next(atom for atom in modified.atoms if atom.name == "P")
    distorted_sidechain = trajectory[:]
    distorted_sidechain.xyz[:, phosphorus.index, 0] += 0.5
    sidechain_stats = compute_physicality_stats(distorted_sidechain)
    assert (sidechain_stats["bad_bonds_sidechain"] > baseline["bad_bonds_sidechain"]).all()

    previous_carbon = next(atom for atom in previous.atoms if atom.name == "C")
    modified_nitrogen = next(atom for atom in modified.atoms if atom.name == "N")
    aligned = trajectory[:]
    aligned.xyz[:, previous_carbon.index] = aligned.xyz[:, modified_nitrogen.index] + np.array(
        [0.13, 0, 0]
    )
    aligned_stats = compute_physicality_stats(aligned)
    distorted_peptide = aligned[:]
    distorted_peptide.xyz[:, previous_carbon.index, 0] += 0.5
    peptide_stats = compute_physicality_stats(distorted_peptide)
    assert (peptide_stats["bad_bonds_backbone"] > aligned_stats["bad_bonds_backbone"]).all()


def test_unmodified_placeholder_uses_unknown_template():
    sequence = "AXXS"
    structure = inference_input_structure_from_protein_sequence(
        sequence,
        template_cache=load_polymer_templates(),
        modifications=[(2, "ccd", "NLE")],
    )
    assert structure.sequence == sequence
    labels_by_residue = {
        residue_idx: residue_label
        for residue_idx, residue_label in zip(
            structure.residue_idx.cpu().tolist(), structure.residue_labels, strict=True
        )
    }
    assert [labels_by_residue[idx] for idx in range(4)] == ["ALA", "NLE", "UNK", "SER"]


def test_system_modification_on_canonical_residue_is_converted_by_position():
    protein = Protein(
        sequence="MATXX",
        asym_ids="A",
        mods=[
            ModifiedResidue(res_id=3, ccd_code="TPO"),
            ModifiedResidue(res_id=4, ccd_code="NLE"),
        ],
    )

    structure = AllAtomStructure.from_system(System(entities=[protein]))
    residue_labels = dict(
        zip(structure.residue_idx.cpu().tolist(), structure.residue_labels, strict=True)
    )

    assert structure.sequence == protein.sequence == "MATXX"
    assert [residue_labels[position] for position in range(5)] == [
        "MET",
        "ALA",
        "TPO",
        "NLE",
        "UNK",
    ]
    msa, _, _ = tokenize_msa(structure, unpaired_a3ms={0: ">query\nMATXX\n"})
    assert msa.shape[0] == structure.num_tokens
