"""Tests for branched entity (glycan) support.

Covers:
- ``template_from_cifwrapper`` with ``remove_leaving_atoms`` as bool
- ``is_leaving_atom_donor`` helper
- ``inference_input_structure_from_branched_entity``
- ``_get_branched_structure`` with bond-driven donor detection

All tests use a local NAG CIF fixture — no internet download required.
"""

import logging
from pathlib import Path

import pytest
from pdbecif.mmcif_io import CifFileReader

from bioemu2.conditioning.contact import contact_conditioning_info
from bioemu2.data_interface import (
    Atom,
    Bond,
    Branched,
    Contact,
    ContactToken,
    ModifiedResidue,
    System,
)
from bioemu2.data_interface.const import MolType
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.all_atom_structure import _get_branched_structure
from bioemu2.structure.inference_inputs import (
    inference_input_structure_from_branched_entity,
    inference_input_structure_from_protein_sequence,
)
from bioemu2.structure.templates import (
    is_leaving_atom_donor,
    load_polymer_templates,
    template_from_cifwrapper,
)

NAG_CIF_PATH = Path(__file__).parent / "test_nag.cif"


@pytest.fixture()
def nag_cifwrapper():
    """Load the local NAG CIF fixture as a CIFWrapper."""
    cfr = CifFileReader(input="data")
    cif_dict = cfr.read(str(NAG_CIF_PATH), output="cif_wrapper")
    return list(cif_dict.values())[0]


class TestTemplateFromCifwrapper:
    def test_no_removal(self, nag_cifwrapper):
        """Default: all heavy atoms kept (15 for NAG)."""
        t = template_from_cifwrapper(nag_cifwrapper, remove_leaving_atoms=False)
        assert t.num_nodes == 15  # NAG has 15 heavy atoms

    def test_unconditional_removal(self, nag_cifwrapper):
        """True: O1 (the only heavy leaving atom) removed → 14 atoms."""
        t = template_from_cifwrapper(nag_cifwrapper, remove_leaving_atoms=True)
        assert t.num_nodes == 14

    def test_bond_indices_valid_after_removal(self, nag_cifwrapper):
        """Bond indices should reference valid atom positions after filtering."""
        t = template_from_cifwrapper(nag_cifwrapper, remove_leaving_atoms=True)
        assert t.bond_index.max() < t.num_nodes
        assert t.bond_index.min() >= 0
        assert t.bond_index.shape[1] == t.bond_type.shape[0]

    @pytest.mark.parametrize("remove_leaving_atoms", [False, True])
    @pytest.mark.parametrize("leaving_atoms", [(), ("O1",), ("O1", "O4")])
    def test_warning_counts_only_removed_heavy_atoms(
        self, nag_cifwrapper, caplog, remove_leaving_atoms, leaving_atoms
    ):
        atom_block = nag_cifwrapper._chem_comp_atom
        atom_block.pdbx_leaving_atom_flag[:] = [
            "Y" if element == "H" or name in leaving_atoms else "N"
            for name, element in zip(atom_block.atom_id, atom_block.type_symbol, strict=True)
        ]
        with caplog.at_level(logging.WARNING, logger="bioemu2.structure.templates"):
            template = template_from_cifwrapper(
                nag_cifwrapper, remove_leaving_atoms=remove_leaving_atoms
            )

        removed_count = len(leaving_atoms) if remove_leaving_atoms else 0
        assert template.num_nodes == 15 - removed_count
        for name in ("O1", "O4"):
            assert template.name_encoding_is_atom(name).any().item() == (
                not remove_leaving_atoms or name not in leaving_atoms
            )
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "bioemu2.structure.templates"
        ]
        if removed_count > 1:
            assert messages == [
                "Removing 2 heavy leaving atoms (O1, O4) for component NAG according to "
                "pdbx_leaving_atom_flag. This may result in incomplete templates."
            ]
        else:
            assert messages == []


@pytest.mark.parametrize(
    "atom_name, expected",
    [
        ("C1", True),  # glycosidic bond donor
        ("O1", False),  # glycosidic bond acceptor (leaving atom)
        ("O4", False),  # non-leaving atom
    ],
)
def test_is_leaving_atom_donor(atom_name, expected, mock_template_from_ccd):
    """Test is_leaving_atom_donor with different atom names."""
    assert is_leaving_atom_donor("NAG", [atom_name]) is expected


@pytest.fixture()
def mock_template_from_ccd(monkeypatch, nag_cifwrapper):
    """Monkeypatch _load_ccd_cifwrapper to use the local NAG fixture.

    Only supports ccd_code='NAG'; raises for anything else.
    """

    def _patched_load(ccd_code):
        if ccd_code != "NAG":
            raise ValueError(f"Mock only supports NAG, got {ccd_code}")
        return nag_cifwrapper

    monkeypatch.setattr("bioemu2.structure.templates._load_ccd_cifwrapper", _patched_load)
    monkeypatch.setattr("bioemu2.structure.inference_inputs._load_ccd_cifwrapper", _patched_load)


class TestInferenceInputStructureFromBranchedEntity:
    def test_no_removal(self, mock_template_from_ccd):
        s = inference_input_structure_from_branched_entity(["NAG", "NAG"])
        assert s.num_nodes == 30  # 15 + 15
        assert s.num_residues == 2
        assert s.num_tokens == 30  # per-atom tokenization

    def test_donor_removal(self, mock_template_from_ccd):
        """First residue is donor, second is acceptor."""
        s = inference_input_structure_from_branched_entity(
            ["NAG", "NAG"], remove_leaving_atoms=[True, False]
        )
        assert s.num_nodes == 29  # 14 (donor) + 15 (acceptor)

    def test_bool_removal(self, mock_template_from_ccd):
        s = inference_input_structure_from_branched_entity(
            ["NAG", "NAG"], remove_leaving_atoms=[True, False]
        )
        assert s.num_nodes == 29

    def test_all_bonds_intra_residue(self, mock_template_from_ccd):
        """No inter-residue bonds should be present."""
        s = inference_input_structure_from_branched_entity(["NAG", "NAG"])
        for i in range(s.bond_index.shape[1]):
            a1, a2 = s.bond_index[:, i]
            assert s.residue_idx[a1] == s.residue_idx[a2], (
                f"Bond {i} crosses residues: {s.residue_idx[a1]} != {s.residue_idx[a2]}"
            )

    def test_mol_type_nonpolymer(self, mock_template_from_ccd):
        s = inference_input_structure_from_branched_entity(["NAG"])
        assert (s.mol_type == MolType.NONPOLYMER.value).all()

    def test_length_mismatch_raises(self, mock_template_from_ccd):
        with pytest.raises(ValueError, match="remove_leaving_atoms length"):
            inference_input_structure_from_branched_entity(
                ["NAG", "NAG"], remove_leaving_atoms=[True]
            )

    def test_empty_raises(self, mock_template_from_ccd):
        with pytest.raises(ValueError, match="non-empty"):
            inference_input_structure_from_branched_entity([])


def test_contacts_select_atom_within_branched_residue(mock_template_from_ccd):
    system = System(
        entities=[Branched(mon_seq=[ModifiedResidue(ccd_code="NAG")] * 2, asym_ids=["S"])]
    )
    structure = AllAtomStructure.from_system(
        system,
        contacts=[
            Contact(
                token1=ContactToken(asym_id="S", res_id=1, atom_name="C1"),
                token2=ContactToken(asym_id="S", res_id=2, atom_name="O4"),
            )
        ],
    )
    c1 = structure.name_encoding_is_atom("C1")[structure.token_center_atom_index]
    o4 = structure.name_encoding_is_atom("O4")[structure.token_center_atom_index]
    residue_idx = structure.residue_idx[structure.token_center_atom_index]
    c1_index = ((residue_idx == 0) & c1).nonzero().item()
    o4_index = ((residue_idx == 1) & o4).nonzero().item()

    assert (
        structure.contact_conditioning[c1_index, o4_index, contact_conditioning_info["CONTACT"]]
        == 1
    )


class TestGetBranchedStructure:
    def _make_branched(self, n_residues=2):
        return Branched(
            mon_seq=[ModifiedResidue(ccd_code="NAG")] * n_residues,
            asym_ids=["C"],
        )

    def test_no_bonds(self, mock_template_from_ccd):
        """Without bonds, no leaving atoms removed."""
        s = _get_branched_structure(self._make_branched(2), bonds=None)
        assert s.num_nodes == 30

    def test_with_glycosidic_bond(self, mock_template_from_ccd):
        """Glycosidic bond C1(res1)→O4(res2): only res1 is donor."""
        bonds = [
            Bond(
                atom1=Atom(name="C1", res_name="NAG", res_id=1, asym_id="C"),
                atom2=Atom(name="O4", res_name="NAG", res_id=2, asym_id="C"),
            )
        ]
        s = _get_branched_structure(self._make_branched(2), bonds=bonds)
        assert s.num_nodes == 29  # 14 (donor) + 15 (acceptor)

    def test_bond_on_other_chain_ignored(self, mock_template_from_ccd):
        """Bonds referencing a different chain should not affect this entity."""
        bonds = [
            Bond(
                atom1=Atom(name="C1", res_name="NAG", res_id=1, asym_id="X"),
                atom2=Atom(name="O4", res_name="NAG", res_id=2, asym_id="X"),
            )
        ]
        s = _get_branched_structure(self._make_branched(2), bonds=bonds)
        assert s.num_nodes == 30  # no removal


class TestSystemWithBranched:
    """Integration tests converting a full System to AllAtomStructure."""

    @staticmethod
    def _protein_atom_count(sequence: str) -> int:
        """Expected heavy-atom count for a canonical protein sequence."""
        cache = load_polymer_templates()
        return inference_input_structure_from_protein_sequence(sequence, cache).num_nodes

    def test_branched_only_no_bonds(self, mock_template_from_ccd):
        """Branched entity without any bonds → all heavy atoms retained."""
        task = System.model_validate(
            {
                "entities": [
                    {
                        "entity_type": "branched",
                        "mon_seq": [{"ccd_code": "NAG"}, {"ccd_code": "NAG"}],
                        "asym_ids": ["B"],
                    },
                ],
                "bonds": [],
            }
        )
        structure = AllAtomStructure.from_system(task)

        assert structure.num_nodes == 30  # 15 + 15, no removal
        # Single chain → asym_id should be constant
        assert structure.asym_id.unique().numel() == 1

    def test_protein_plus_branched_with_glycosylation(self, mock_template_from_ccd):
        """Protein + branched with intra-branched + protein-to-branched glycosylation bonds.

        Bonds:
        - NAG(1) C1 → NAG(2) O4  (intra-branched glycosidic bond)
        - Protein ASN ND2 → NAG(1) C1 on chain B (N-linked glycosylation)

        NAG(1) has C1 in TWO bonds. Both are donor-side → O1 removed from NAG(1).
        NAG(2) only appears as acceptor (O4 side) → no removal.
        """
        seq = "ACNE"  # N at position 3 (res_id=3)
        task = System.model_validate(
            {
                "entities": [
                    {"entity_type": "protein", "sequence": seq, "asym_ids": ["A"]},
                    {
                        "entity_type": "branched",
                        "mon_seq": [{"ccd_code": "NAG"}, {"ccd_code": "NAG"}],
                        "asym_ids": ["B"],
                    },
                ],
                "bonds": [
                    {
                        "atom1": {"name": "O4", "res_name": "NAG", "res_id": 1, "asym_id": "B"},
                        "atom2": {"name": "C1", "res_name": "NAG", "res_id": 2, "asym_id": "B"},
                    },
                    {
                        "atom1": {"name": "ND2", "res_name": "ASN", "res_id": 3, "asym_id": "A"},
                        "atom2": {"name": "C1", "res_name": "NAG", "res_id": 1, "asym_id": "B"},
                    },
                ],
            }
        )
        structure = AllAtomStructure.from_system(task)

        protein_atoms = self._protein_atom_count(seq)
        branched_atoms = 14 + 14  # NAG(1) and NAG(2) donor (O1 removed)
        expected_total = protein_atoms + branched_atoms
        assert structure.num_nodes == expected_total

        # Extra bonds should have been added by _add_extra_covalent_bonds
        # Branched has intra-residue, and there are 2 cross-residue bonds (glycosylation linkage +
        # glycosidic bond) → total 30 bonds
        assert structure.bond_index.shape[1] == 14 * 2 + 2
        # Check the 2 extra bonds are present (total bonds = intra-chain + 2)
        # We verify by checking that at least some bonds cross the asym_id boundary.
        asym_at_atom = structure.upsample_to_atoms(structure.asym_id)
        cross_chain_bonds = 0
        for i in range(structure.bond_index.shape[1]):
            a1, a2 = structure.bond_index[:, i]
            if asym_at_atom[a1] != asym_at_atom[a2]:
                cross_chain_bonds += 1
        assert cross_chain_bonds >= 1, "Expected at least one cross-chain bond (glycosylation)"
