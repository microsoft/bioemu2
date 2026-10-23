"""Tests for CIF writing utilities."""

from pathlib import Path

import mdtraj
import pytest
import torch
from biotite.structure.io import pdbx

from bioemu2.data_interface import System
from bioemu2.structure import AllAtomStructure, get_asym_id_to_chain_metadata
from bioemu2.structure.cif import save_structure_to_cif


def _make_system(task_dict: dict) -> System:
    return System.model_validate(task_dict)


def _save_and_read_cif(system: System, out_path: Path) -> pdbx.CIFBlock:
    """Build structure from System, write CIF via save_structure_to_cif, return block."""
    asym_id_to_chain_metadata = get_asym_id_to_chain_metadata(system)
    renamed_asym_ids = list(asym_id_to_chain_metadata.keys())
    structure = AllAtomStructure.from_system(system)
    structure = structure.replace(coords=torch.zeros(structure.num_nodes, 3))
    save_structure_to_cif(
        structure,
        out_path,
        renamed_asym_ids=renamed_asym_ids,
        system=system,
        asym_id_to_chain_metadata=asym_id_to_chain_metadata,
    )
    cif_file = pdbx.CIFFile.read(str(out_path))
    return cif_file[next(iter(cif_file.keys()))]


class TestCifProteinOnly:
    """Test CIF output for a protein-only system."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.task = _make_system(
            {
                "entities": [
                    {"entity_type": "protein", "sequence": "ACDEF", "asym_ids": ["A"]},
                ],
            }
        )
        self.block = _save_and_read_cif(self.task, tmp_path / "topology.cif")

    def test_entity_table_present(self):
        assert "entity" in self.block

    def test_entity_ids(self):
        assert self.block["entity"]["id"].as_array(str).tolist() == ["1"]

    def test_entity_type_is_polymer(self):
        assert self.block["entity"]["type"].as_array(str).tolist() == ["polymer"]

    def test_entity_poly_table(self):
        assert "entity_poly" in self.block
        assert self.block["entity_poly"]["entity_id"].as_array(str).tolist() == ["1"]
        assert self.block["entity_poly"]["type"].as_array(str).tolist() == ["polypeptide(L)"]
        assert self.block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str).tolist() == [
            "ACDEF"
        ]

    def test_struct_asym_table(self):
        assert "struct_asym" in self.block
        assert self.block["struct_asym"]["id"].as_array(str).tolist() == ["A"]
        assert self.block["struct_asym"]["entity_id"].as_array(str).tolist() == ["1"]

    def test_label_entity_id_patched(self):
        entity_ids = self.block["atom_site"]["label_entity_id"].as_array(str).tolist()
        assert all(eid == "1" for eid in entity_ids)
        assert "?" not in entity_ids


class TestCifProteinWithLigand:
    """Test CIF output for a protein + ligand system."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.task = _make_system(
            {
                "entities": [
                    {
                        "entity_type": "protein",
                        "sequence": "ACDEF",
                        "asym_ids": ["A1"],
                    },
                    {
                        "entity_type": "ligand",
                        "ccd_code": "ZN",
                        "asym_ids": ["E1", "F1"],
                    },
                ],
            }
        )
        self.block = _save_and_read_cif(self.task, tmp_path / "topology.cif")

    def test_entity_table_has_two_entities(self):
        assert self.block["entity"]["id"].as_array(str).tolist() == ["1", "2"]
        assert self.block["entity"]["type"].as_array(str).tolist() == ["polymer", "non-polymer"]

    def test_entity_poly_only_contains_polymer(self):
        assert self.block["entity_poly"]["entity_id"].as_array(str).tolist() == ["1"]
        assert self.block["entity_poly"]["type"].as_array(str).tolist() == ["polypeptide(L)"]

    def test_struct_asym_maps_all_chains(self):
        assert self.block["struct_asym"]["id"].as_array(str).tolist() == ["A1", "E1", "F1"]
        assert self.block["struct_asym"]["entity_id"].as_array(str).tolist() == ["1", "2", "2"]
        site = self.block["atom_site"]
        asym = site["label_asym_id"].as_array(str)
        seq = site["label_seq_id"].as_array(str)
        assert {chain: list(dict.fromkeys(seq[asym == chain])) for chain in ("A1", "E1", "F1")} == {
            "A1": list(map(str, range(1, 6))),
            "E1": ["1"],
            "F1": ["1"],
        }

    def test_label_entity_id_correct_per_chain(self):
        asym_ids = self.block["atom_site"]["label_asym_id"].as_array(str).tolist()
        entity_ids = self.block["atom_site"]["label_entity_id"].as_array(str).tolist()
        for asym_id, entity_id in zip(asym_ids, entity_ids):
            if asym_id == "A1":
                assert entity_id == "1"
            elif asym_id in ("E1", "F1"):
                assert entity_id == "2"
            else:
                pytest.fail(f"Unexpected asym_id: {asym_id}")

    def test_ccd_ligand_residue_name_retained(self):
        asym_ids = self.block["atom_site"]["label_asym_id"].as_array(str).tolist()
        comp_ids = self.block["atom_site"]["label_comp_id"].as_array(str).tolist()
        assert {comp for asym_id, comp in zip(asym_ids, comp_ids) if asym_id in ("E1", "F1")} == {
            "ZN"
        }


@pytest.mark.parametrize("unknown_label", ["UNK", "FAKE"])
def test_cif_mixed_ccd_and_non_ccd_ligands(tmp_path: Path, unknown_label: str):
    task = _make_system(
        {"entities": [{"entity_type": "ligand", "ccd_code": "ZN", "asym_ids": ["A", "B"]}]}
    )
    structure = AllAtomStructure.from_system(task)
    labels = list(structure.residue_labels)
    labels[1] = unknown_label
    structure = structure.replace(coords=torch.zeros(structure.num_nodes, 3), residue_labels=labels)
    path = tmp_path / "topology.cif"

    save_structure_to_cif(structure, path, renamed_asym_ids=["A", "B"])
    block = pdbx.CIFFile.read(str(path))["topology"]
    atom_site = block["atom_site"]
    assert atom_site["label_asym_id"].as_array(str).tolist() == ["A", "B"]
    assert atom_site["label_comp_id"].as_array(str).tolist() == ["ZN", "LIG"]

    save_structure_to_cif(structure, path, renamed_asym_ids=["A", "B"], rename_residues={})
    block = pdbx.CIFFile.read(str(path))["topology"]
    assert block["atom_site"]["label_comp_id"].as_array(str).tolist() == ["ZN", "LIG"]

    save_structure_to_cif(structure, path, renamed_asym_ids=["A", "B"], rename_residues={0: "CU"})
    block = pdbx.CIFFile.read(str(path))["topology"]
    assert block["atom_site"]["label_comp_id"].as_array(str).tolist() == ["CU", "LIG"]

    save_structure_to_cif(structure, path, renamed_asym_ids=["A", "B"], rename_residues={1: "HEM"})
    block = pdbx.CIFFile.read(str(path))["topology"]
    assert block["atom_site"]["label_comp_id"].as_array(str).tolist() == ["ZN", "HEM"]


def test_cif_writes_model_formal_charges_for_all_atoms(tmp_path: Path) -> None:
    system = System.from_description(
        proteins=["A"],
        ligand_smiles=["[Na+]", "CC(=O)NCC(=O)OP(=O)([O-])O"],
    )
    structure = AllAtomStructure.from_system(system)
    path = tmp_path / "charged.cif"
    save_structure_to_cif(structure, path, renamed_asym_ids=system.chain_ids)

    atom_site = pdbx.CIFFile.read(str(path)).block["atom_site"]
    charges = atom_site["pdbx_formal_charge"].as_array(str).tolist()
    chains = atom_site["label_asym_id"].as_array(str).tolist()
    elements = atom_site["type_symbol"].as_array(str).tolist()

    assert charges == [str(int(charge)) for charge in structure.charge.tolist()]
    assert {charge for charge, chain in zip(charges, chains) if chain == "A"} == {"0"}
    assert [
        (element, charge)
        for element, charge, chain in zip(elements, charges, chains)
        if chain == "B"
    ] == [("Na", "1")]
    assert [
        (element, charge)
        for element, charge, chain in zip(elements, charges, chains)
        if chain == "C" and charge != "0"
    ] == [("O", "-1")]


@pytest.mark.parametrize("charge", [0.5, float("nan")])
def test_cif_rejects_noninteger_or_nonfinite_formal_charge(tmp_path: Path, charge: float) -> None:
    system = System.from_description(ligand_smiles=["[Na+]"])
    structure = AllAtomStructure.from_system(system)
    structure = structure.replace(charge=torch.tensor([charge], dtype=torch.float32))

    with pytest.raises(ValueError, match="finite integers"):
        save_structure_to_cif(structure, tmp_path / "invalid.cif")


@pytest.mark.parametrize("ccd_codes", [["ATP"], ["ZN", "ATP"]])
@pytest.mark.parametrize("protein_chains", [1, 2])
def test_ccd_ligands_survive_system_cif_roundtrip(
    tmp_path: Path, ccd_codes: list[str], protein_chains: int
):
    system = System.from_description(
        proteins=["ACDEF"] * protein_chains, ligand_ccd_codes=ccd_codes
    )
    path = tmp_path / "topology.cif"
    _save_and_read_cif(system, path)

    inferred = System.from_cif(path)

    assert inferred.entities == system.entities
    assert inferred.topology == system.topology
    assert mdtraj.load(str(path)).n_atoms == inferred.topology.n_atoms


class TestCifBranchedEntity:
    """Test mmCIF metadata for a branched NAG entity alongside a protein."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        def nag_ccd_path(ccd_code: str) -> Path:
            if ccd_code != "NAG":
                raise ValueError(f"Unexpected CCD code: {ccd_code}")
            return Path(__file__).parent / "test_nag.cif"

        monkeypatch.setattr("bioemu2.structure.templates._get_ccd_path", nag_ccd_path)
        self.task = _make_system(
            {
                "entities": [
                    {
                        "entity_type": "protein",
                        "sequence": "ACDEF",
                        "asym_ids": ["A"],
                    },
                    {
                        "entity_type": "branched",
                        "asym_ids": ["B"],
                        "mon_seq": [{"ccd_code": "NAG"}, {"ccd_code": "NAG"}],
                    },
                ],
            }
        )
        self.block = _save_and_read_cif(self.task, tmp_path / "topology.cif")

    def test_entity_type_is_branched(self):
        assert self.block["entity"]["type"].as_array(str).tolist() == ["polymer", "branched"]

    def test_branched_residue_labels_keep_ccd_codes(self):
        asym_ids = self.block["atom_site"]["label_asym_id"].as_array(str).tolist()
        comp_ids = self.block["atom_site"]["label_comp_id"].as_array(str).tolist()
        assert {comp for asym_id, comp in zip(asym_ids, comp_ids) if asym_id == "B"} == {"NAG"}
        seq = self.block["atom_site"]["label_seq_id"].as_array(str).tolist()
        assert list(dict.fromkeys(n for a, n in zip(asym_ids, seq) if a == "B")) == ["1", "2"]


def test_modified_protein_bonds_survive_cif_roundtrip(tmp_path: Path):
    system = _make_system(
        {
            "entities": [
                {
                    "entity_type": "protein",
                    "sequence": "AXA",
                    "asym_ids": ["A"],
                    "mods": [{"res_id": 2, "ccd_code": "TPO"}],
                }
            ]
        }
    )
    path = tmp_path / "topology.cif"
    _save_and_read_cif(system, path)
    topology = mdtraj.load(str(path)).topology
    previous, modified, following = list(topology.residues)
    bonds = {frozenset((a.index, b.index)) for a, b in topology.bonds}

    assert any(a.residue == modified and b.residue == modified for a, b in topology.bonds)
    for first, second in ((previous, modified), (modified, following)):
        carbon = next(atom for atom in first.atoms if atom.name == "C")
        nitrogen = next(atom for atom in second.atoms if atom.name == "N")
        assert frozenset((carbon.index, nitrogen.index)) in bonds


class TestCifMultichain:
    """Test CIF output for a homodimer + RNA system."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.task = _make_system(
            {
                "entities": [
                    {
                        "entity_type": "protein",
                        "sequence": "GGG",
                        "asym_ids": ["A", "B"],
                    },
                    {
                        "entity_type": "rna",
                        "sequence": "GCAU",
                        "asym_ids": ["C"],
                    },
                ],
            }
        )
        self.block = _save_and_read_cif(self.task, tmp_path / "topology.cif")

    def test_entity_table(self):
        assert self.block["entity"]["id"].as_array(str).tolist() == ["1", "2"]
        assert self.block["entity"]["type"].as_array(str).tolist() == ["polymer", "polymer"]

    def test_entity_poly_has_both_polymers(self):
        poly = self.block["entity_poly"]
        assert poly["entity_id"].as_array(str).tolist() == ["1", "2"]
        assert poly["type"].as_array(str).tolist() == ["polypeptide(L)", "polyribonucleotide"]
        assert poly["pdbx_seq_one_letter_code_can"].as_array(str).tolist() == ["GGG", "GCAU"]

    def test_struct_asym(self):
        asym = self.block["struct_asym"]
        assert asym["id"].as_array(str).tolist() == ["A", "B", "C"]
        assert asym["entity_id"].as_array(str).tolist() == ["1", "1", "2"]
        site = self.block["atom_site"]
        chain_ids = site["label_asym_id"].as_array(str)
        seq = site["label_seq_id"].as_array(str)
        assert {
            chain: list(dict.fromkeys(seq[chain_ids == chain])) for chain in ("A", "B", "C")
        } == {"A": ["1", "2", "3"], "B": ["1", "2", "3"], "C": ["1", "2", "3", "4"]}
        assert [[r.resSeq for r in c.residues] for c in self.task.topology.chains] == [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3, 4],
        ]
        assert [r.index for r in self.task.topology.residues] == list(range(10))

    def test_label_entity_id_correct(self):
        asym_ids = self.block["atom_site"]["label_asym_id"].as_array(str).tolist()
        entity_ids = self.block["atom_site"]["label_entity_id"].as_array(str).tolist()
        for asym_id, entity_id in zip(asym_ids, entity_ids):
            if asym_id in ("A", "B"):
                assert entity_id == "1"
            elif asym_id == "C":
                assert entity_id == "2"
