import gzip
import logging
from itertools import islice
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from bioemu2.data_interface import (
    DNA,
    RNA,
    Atom,
    Bond,
    Branched,
    Contact,
    ContactToken,
    Ligand,
    ModifiedResidue,
    Protein,
    System,
    load_system_and_contacts,
)
from bioemu2.data_interface.system import _asym_ids

_TEST_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' ACDE
#
loop_
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.comp_id
2 EDO
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
A 1 A
B 2 B
C 2 C
#
"""

_MODIFIED_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' ACDE
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
A 1 A
#
loop_
_pdbx_struct_mod_residue.label_asym_id
_pdbx_struct_mod_residue.label_comp_id
_pdbx_struct_mod_residue.label_seq_id
A SEP 2
#
"""

_ASSEMBLY_MODIFIED_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
2 'polypeptide(L)' RVIYFVPLNR
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
2 1 ARG
2 2 VAL
2 3 ILE
2 4 PTR
2 5 PHE
2 6 VAL
2 7 PRO
2 8 LEU
2 9 ASN
2 10 ARG
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
B 4 PTR
C 4 PTR
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
B 2 P
C 2 Q
#
"""

_EXPANDED_MODIFIED_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_seq_one_letter_code
1 'polypeptide(L)' ATYGD 'A(CRO)D'
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 CRO
1 3 ASP
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
A 1 ALA
A 2 CRO
A 3 ASP
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
A 1 A
#
"""

_THYMINE_DIMER_CIF = """\
data_TEST
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_seq_one_letter_code
1 'polydeoxyribonucleotide' ATCGGCTTCGCGCA '(DA)(DT)(DC)(DG)(DG)(DC)(WUH)(DC)(DG)(DC)(DG)(DC)(DA)'
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 DA
1 2 DT
1 3 DC
1 4 DG
1 5 DG
1 6 DC
1 7 WUH
1 8 DC
1 9 DG
1 10 DC
1 11 DG
1 12 DC
1 13 DA
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
C 1 C
E 1 E
#
"""

_AMBIGUOUS_AUTHOR_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' ACDE
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
A 1 E
B 1 E
#
"""

_UNSUPPORTED_CIF = """\
data_TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' ACDE
2 'polysaccharide(D)' '(NAG)'
#
loop_
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.auth_asym_id
A 1 X
B 2 Y
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
connection1 covale
#
"""

_MINIMAL_SYSTEM_YAML = {"entities": [{"entity_type": "protein", "sequence": "AC", "asym_ids": "A"}]}
_CONTACT_YAML = {
    "token1": {"asym_id": "A", "res_id": 1},
    "token2": {"asym_id": "A", "res_id": 2},
}


@pytest.fixture
def prot():
    return Protein(
        sequence="MADCLTCEQXAEFKEAFSLF",
        asym_ids="A",
        mods=[ModifiedResidue(res_id=10, ccd_code="SEP")],
    )


@pytest.fixture
def prot_w_lengthy_asym_id():
    return Protein(
        sequence="MADCLTCEQXAEFKEAFSLF",
        asym_ids="ABCDE",
        mods=[ModifiedResidue(res_id=10, ccd_code="SEP")],
    )


@pytest.fixture
def rna():
    return RNA(sequence="GCAUAGC", asym_ids="C")


@pytest.fixture
def dna():
    return DNA(sequence="GCAAGC", asym_ids="D")


@pytest.fixture
def ligand_smiles():
    return Ligand(smiles="CC1=CC=CC=C1", asym_ids="lig1")


@pytest.fixture
def ligand_ccd():
    return Ligand(ccd_code="ATP", asym_ids="lig2")


@pytest.fixture
def ligand_inchi():
    return Ligand(inchi="InChI=1S/CH4/h1H4", asym_ids="lig3")


@pytest.fixture
def cif_path(tmp_path: Path) -> Path:
    path = tmp_path / "structure.cif"
    path.write_text(_TEST_CIF)
    return path


def test_system_instantiation(prot, rna, dna, ligand_smiles, ligand_ccd, ligand_inchi):
    system = System(
        entities=[prot, rna, dna, ligand_smiles, ligand_ccd, ligand_inchi],
        bonds=[
            Bond(
                atom1=Atom(name="SG", res_name="CYS", res_id=4, asym_id="A"),
                atom2=Atom(name="SG", res_name="CYS", res_id=7, asym_id="A"),
            ),
            Bond(
                atom1=Atom(name="N1", res_name="G", res_id=1, asym_id="C"),
                atom2=Atom(name="O6", res_name="G", res_id=6, asym_id="C"),
            ),
        ],
    )

    assert isinstance(system, System)
    assert len(system.entities) == 6
    assert system.bonds is not None
    assert len(system.bonds) == 2


def test_system_accepts_positional_entities(prot):
    entities = [prot]
    positional = System(entities, asym_id_to_author_id={"A": "X"})
    keyword = System(entities=entities, asym_id_to_author_id={"A": "X"})

    assert positional == keyword
    assert positional.chain_ids == ["A"]
    assert positional.bonds == []
    assert positional.asym_id_to_author_id == {"A": "X"}
    assert System.model_validate_json(positional.model_dump_json()) == positional


def test_system_positional_entities_are_validated():
    with pytest.raises(ValidationError, match="List should have at least 1 item"):
        System([])
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'entities'"):
        System.model_validate({})
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'entities'"):
        System.model_validate_json("{}")


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(
            {
                "entity_type": "protein",
                "sequence": "AX",
                "asym_ids": ["A"],
                "mods": [{"ccd_code": "SEP"}],
            },
            id="protein",
        ),
        pytest.param(
            {
                "entity_type": "rna",
                "sequence": "AN",
                "asym_ids": ["A"],
                "mods": [{"ccd_code": "4AC"}],
            },
            id="rna",
        ),
        pytest.param(
            {
                "entity_type": "dna",
                "sequence": "AN",
                "asym_ids": ["A"],
                "mods": [{"ccd_code": "D5M"}],
            },
            id="dna",
        ),
    ],
)
def test_system_from_yaml_rejects_unpositioned_polymer_modifications(entity, tmp_path: Path):
    path = tmp_path / "unpositioned-system.yaml"
    path.write_text(yaml.safe_dump({"entities": [entity]}))
    with pytest.raises(ValueError, match="Every .* modification must include a 1-based res_id"):
        System.from_yaml(path)


def test_polymer_modifications_use_residue_positions():
    protein = Protein(
        sequence="ASX",
        asym_ids="A",
        mods=[ModifiedResidue(res_id=2, ccd_code="SEP")],
    )
    assert protein.residue_sequence == ["ALA", "SEP", "UNK"]


@pytest.mark.parametrize(
    "mods",
    [
        pytest.param([ModifiedResidue(ccd_code="SEP")], id="missing-res-id"),
        pytest.param(
            [
                ModifiedResidue(res_id=2, ccd_code="SEP"),
                ModifiedResidue(res_id=2, ccd_code="TPO"),
            ],
            id="duplicate-res-id",
        ),
        pytest.param([ModifiedResidue(res_id=4, ccd_code="SEP")], id="out-of-range"),
    ],
)
def test_polymer_modification_positions_are_validated(mods):
    with pytest.raises(ValueError):
        Protein(sequence="ASX", asym_ids="A", mods=mods)


@pytest.mark.parametrize("field", ["contacts", "mode"])
def test_system_rejects_conditioning_fields(prot, field):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        System.model_validate(
            {"entities": [prot], field: [] if field == "contacts" else "equilibrium"}
        )


def _write_system_yaml(tmp_path: Path, **fields: object) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(_MINIMAL_SYSTEM_YAML | fields, sort_keys=False))
    return path


def test_system_from_yaml_ignores_conditioning_fields(tmp_path: Path) -> None:
    system = System.from_yaml(
        _write_system_yaml(
            tmp_path,
            mode="not-a-mode",
            contacts="not-contacts",
        )
    )

    assert system == System(entities=[Protein(sequence="AC", asym_ids=["A"])])


def test_system_from_yaml_rejects_other_extra_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        System.from_yaml(_write_system_yaml(tmp_path, unknown="value"))


@pytest.mark.parametrize(
    ("fields", "expected_contacts"),
    [
        pytest.param({}, None, id="without-contacts"),
        pytest.param({"contacts": None}, None, id="null-contacts"),
        pytest.param({"contacts": []}, [], id="empty-contacts"),
        pytest.param(
            {"contacts": [_CONTACT_YAML]},
            [Contact.model_validate(_CONTACT_YAML)],
            id="with-contacts",
        ),
    ],
)
def test_load_system_and_contacts_from_flat_yaml(
    tmp_path: Path,
    fields: dict[str, object],
    expected_contacts: list[Contact] | None,
) -> None:
    system, contacts = load_system_and_contacts(_write_system_yaml(tmp_path, **fields))

    assert system == System(entities=[Protein(sequence="AC", asym_ids=["A"])])
    assert contacts == expected_contacts


@pytest.mark.parametrize("fields", [{"unknown": "value"}, {"mode": "structure"}])
def test_load_system_and_contacts_rejects_unknown_yaml_fields(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="Unknown fields"):
        load_system_and_contacts(_write_system_yaml(tmp_path, **fields))


def test_load_system_and_contacts_defaults_ligand_res_id(tmp_path: Path) -> None:
    fields = {
        "entities": [
            {"entity_type": "protein", "sequence": "AC", "asym_ids": "A"},
            {"entity_type": "ligand", "smiles": "CC", "asym_ids": "L"},
        ],
        "contacts": [
            {"token1": {"asym_id": "A", "res_id": 1}, "token2": {"asym_id": "L", "atom_name": "C1"}}
        ],
    }
    system, contacts = load_system_and_contacts(_write_system_yaml(tmp_path, **fields))

    assert system.chain_ids == ["A", "L"]
    assert contacts == [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="L", atom_name="C1"),
        )
    ]


def test_load_system_and_contacts_validates_contacts(tmp_path: Path) -> None:
    bad_contact = {"token1": {"asym_id": "A", "res_id": 1}, "token2": {"asym_id": "Z", "res_id": 2}}
    with pytest.raises(ValueError, match="not present in any entity"):
        load_system_and_contacts(_write_system_yaml(tmp_path, contacts=[bad_contact]))


def test_author_id_mapping_uses_canonical_asym_id_order():
    system = System(
        entities=[Protein(sequence="ACDE", asym_ids=["B", "A"])],
        asym_id_to_author_id={"A": "Y", "B": "X"},
    )
    assert list(system.asym_id_to_author_id) == system.chain_ids == ["B", "A"]
    assert list(system.asym_id_to_author_id) == system.chain_ids == ["B", "A"]


def test_mdtraj_conversions() -> None:
    system = System(
        entities=[
            Protein(sequence="GYD", asym_ids=["H"]),
            Protein(sequence="PET", asym_ids=["L"]),
        ]
    )
    topology = system.topology

    assert [chain.chain_id for chain in topology.chains] == ["H", "L"]
    assert system.chain_ids == ["H", "L"]
    assert system.topology is not system.topology


def test_lengthy_asym_id(prot_w_lengthy_asym_id):
    with pytest.raises(ValueError, match="long"):
        System(entities=[prot_w_lengthy_asym_id])


def test_invalid_bonds(prot):
    invalid_bonds = [
        # atom1 res_id out of range
        Bond(
            atom1=Atom(name="SG", res_name="CYS", res_id=77, asym_id="A"),
            atom2=Atom(name="SG", res_name="CYS", res_id=7, asym_id="A"),
        ),
        # atom2 asym_id not in entities
        Bond(
            atom1=Atom(name="SG", res_name="CYS", res_id=4, asym_id="A"),
            atom2=Atom(name="SG", res_name="CYS", res_id=7, asym_id="B"),
        ),
        # atom1 res_name mismatch
        Bond(
            atom1=Atom(name="SG", res_name="TYR", res_id=4, asym_id="A"),
            atom2=Atom(name="SG", res_name="CYS", res_id=7, asym_id="A"),
        ),
    ]

    for bond in invalid_bonds:
        with pytest.raises(ValueError):
            System(entities=[prot], bonds=[bond])


def test_valid_bond_with_modified_residue(prot):
    # a modified residue should be referenced by its ccd code if it has one
    valid_bond = Bond(
        atom1=Atom(name="SG", res_name="CYS", res_id=7, asym_id="A"),
        atom2=Atom(name="O2P", res_name="SEP", res_id=10, asym_id="A"),
    )
    system = System(entities=[prot], bonds=[valid_bond])
    assert isinstance(system, System)
    # and not as "UNK" in this case
    invalid_bond = Bond(
        atom1=Atom(name="SG", res_name="CYS", res_id=7, asym_id="A"),
        atom2=Atom(name="O2P", res_name="UNK", res_id=10, asym_id="A"),
    )
    with pytest.raises(ValueError):
        System(entities=[prot], bonds=[invalid_bond])


def test_contact_accepts_named_and_mapping_endpoints():
    contact = Contact(
        token1=ContactToken(asym_id="A", res_id=5),
        token2={"asym_id": "L", "res_id": 1, "atom_name": "C1"},
    )

    assert contact.token1 == ContactToken(asym_id="A", res_id=5)
    assert contact.token2 == ContactToken(asym_id="L", res_id=1, atom_name="C1")
    serialized = contact.model_dump(mode="json", exclude_none=True)
    assert serialized["token1"] == {"asym_id": "A", "res_id": 5}
    assert serialized["token2"] == {"asym_id": "L", "atom_name": "C1"}
    assert Contact.model_validate(serialized) == contact
    assert Contact.model_validate_json(contact.model_dump_json()) == contact


def test_contact_rejects_tuple_endpoints():
    with pytest.raises(ValidationError, match="valid dictionary or instance of ContactToken"):
        Contact(
            token1=("A", 5),
            token2=ContactToken(asym_id="L", atom_name="C1"),
        )


def test_contact_serialization_preserves_branched_residue_one():
    contact = Contact(
        token1=ContactToken(asym_id="S", res_id=1, atom_name="O4"),
        token2=ContactToken(asym_id="S", res_id=2, atom_name="C1"),
    )
    serialized = contact.model_dump(mode="json", exclude_none=True)

    assert serialized["token1"] == {"asym_id": "S", "atom_name": "O4"}
    assert serialized["token2"] == {"asym_id": "S", "res_id": 2, "atom_name": "C1"}
    assert Contact.model_validate(serialized) == contact


def test_contact_token_requires_explicit_residue_for_residue_endpoint():
    with pytest.raises(ValidationError, match="res_id must be explicitly set"):
        Contact(
            token1={"asym_id": "A"},
            token2=ContactToken(asym_id="A", res_id=2),
        )


@pytest.mark.parametrize(
    "max_distance",
    [pytest.param(3.9, id="below-min"), pytest.param(20.1, id="above-max")],
)
def test_contact_max_distance_out_of_range(max_distance):
    with pytest.raises(ValueError):
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="A", res_id=2),
            max_distance=max_distance,
        )


def test_system_validates_contacts(prot, ligand_smiles):
    contact_res_res = Contact(
        token1=ContactToken(asym_id="A", res_id=5),
        token2=ContactToken(asym_id="A", res_id=15),
        max_distance=8.0,
    )
    contact_res_atom = Contact(
        token1=ContactToken(asym_id="A", res_id=3),
        token2=ContactToken(asym_id="lig1", res_id=1, atom_name="C1"),
    )
    system = System(entities=[prot, ligand_smiles])
    contacts = [contact_res_res, contact_res_atom]

    system.validate_contacts(contacts)

    assert contacts[0].max_distance == 8.0
    assert contacts[1].max_distance == 6.0  # default


def test_contact_unknown_asym_id(prot):
    system = System(entities=[prot])
    with pytest.raises(ValueError, match="not present in any entity"):
        system.validate_contacts(
            [
                Contact(
                    token1=ContactToken(asym_id="A", res_id=5),
                    token2=ContactToken(asym_id="Z", res_id=2),
                )
            ]
        )


def test_contact_res_id_out_of_range(prot):
    system = System(entities=[prot])
    with pytest.raises(ValueError, match="exceeds"):
        system.validate_contacts(
            [
                Contact(
                    token1=ContactToken(asym_id="A", res_id=5),
                    token2=ContactToken(asym_id="A", res_id=999),
                )
            ]
        )


def test_contact_invalid_endpoint_for_entity(prot, ligand_smiles):
    system = System(entities=[prot, ligand_smiles])
    with pytest.raises(ValidationError, match="greater than 0"):
        Contact(
            token1={"asym_id": "A", "res_id": 0},
            token2=ContactToken(asym_id="lig1", atom_name="C1"),
        )
    with pytest.raises(ValueError, match="must specify atom_name"):
        system.validate_contacts(
            [
                Contact(
                    token1=ContactToken(asym_id="A", res_id=5),
                    token2=ContactToken(asym_id="lig1", res_id=1),
                )
            ]
        )
    with pytest.raises(ValueError, match="res_id=1"):
        system.validate_contacts(
            [
                Contact(
                    token1=ContactToken(asym_id="A", res_id=5),
                    token2=ContactToken(asym_id="lig1", res_id=2, atom_name="C1"),
                )
            ]
        )
    with pytest.raises(ValidationError, match="valid integer"):
        Contact(
            token1=ContactToken(asym_id="A", res_id=5),
            token2={"asym_id": "lig1", "res_id": None, "atom_name": "C1"},
        )


def test_contact_branched_endpoint_requires_residue_and_atom():
    system = System(entities=[Branched(asym_ids="S", mon_seq=[ModifiedResidue(ccd_code="NAG")])])
    system.validate_contacts(
        [
            Contact(
                token1=ContactToken(asym_id="S", res_id=1, atom_name="O4"),
                token2=ContactToken(asym_id="S", res_id=1, atom_name="C1"),
            )
        ]
    )
    with pytest.raises(ValueError, match="must specify atom_name"):
        system.validate_contacts(
            [
                Contact(
                    token1=ContactToken(asym_id="S", res_id=1),
                    token2=ContactToken(asym_id="S", res_id=1, atom_name="C1"),
                )
            ]
        )


def test_merge_ligands():
    ligand_a = Ligand(asym_ids="L", smiles="C1=CC=C2C(=C1)C=CC3=C2C=CC=C3")
    ligand_b = Ligand(asym_ids="R", smiles="C1=CC=C2C(=C1)C=CC3=C2C=CC=C3")
    ligand_c = Ligand(asym_ids="T", smiles="C1=CC=C1")
    ligand_d = Ligand(asym_ids="P", inchi="InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H")

    system = System(entities=[ligand_a, ligand_b, ligand_c, ligand_d])
    assert len(system.entities) == 3  # ligand_a and ligand_b should be merged
    merged_ligand = next(
        entity for entity in system.entities if entity.smiles == "C1=CC=C2C(=C1)C=CC3=C2C=CC=C3"
    )
    assert merged_ligand.asym_ids == ["L", "R"]


def test_merge_proteins():
    prot_a = Protein(
        sequence="MADQLTEEQXAEFKEAFSLF",
        asym_ids=["D", "B"],
        mods=[ModifiedResidue(res_id=10, ccd_code="SEP")],
    )
    prot_b = Protein(
        sequence="MADQLTEEQXAEFKEAFSLF",
        asym_ids=["A", "C"],
        mods=[ModifiedResidue(res_id=10, ccd_code="SEP")],
    )
    prot_c = Protein(
        sequence="MADQLTEEQXAEFKEAFSLF",
        asym_ids=["E"],
        mods=[ModifiedResidue(res_id=10, ccd_code="CYX")],
    )
    prot_d = Protein(
        sequence="MADQQTEEQXAEFKEAFSLF",
        asym_ids=["F"],
        mods=[ModifiedResidue(res_id=10, ccd_code="SEP")],
    )

    system = System(entities=[prot_a, prot_b, prot_c, prot_d])
    assert len(system.entities) == 3  # prot_a and prot_b should be merged
    merged_prot = next(
        entity for entity in system.entities if entity.sequence == "MADQLTEEQXAEFKEAFSLF"
    )
    assert merged_prot.asym_ids == ["D", "B", "A", "C"]


def test_entity_merge_does_not_mutate_input():
    """Merged entities must not modify their inputs or prevent reuse."""
    protein = Protein(asym_ids=["A"], sequence="MKTF")
    lig1 = Ligand(asym_ids=["B"], ccd_code="PO4")
    lig2 = Ligand(asym_ids=["C"], ccd_code="PO4")
    entities = [protein, lig1, lig2]

    System(entities=entities)

    # Original entities must be unchanged
    assert protein.asym_ids == ["A"]
    assert lig1.asym_ids == ["B"]
    assert lig2.asym_ids == ["C"]

    # A second construction from the same entities must also succeed
    st2 = System(entities=entities)
    assert len(st2.entities) == 2  # protein + merged ligand


def test_serialization_deserialization(tmpdir, prot, rna, ligand_inchi):
    system = System(
        entities=[prot, rna, ligand_inchi],
    )

    yaml_file = tmpdir.join("system.yaml")
    system.to_yaml(yaml_file)
    assert "contacts" not in yaml_file.read()
    assert "mode" not in yaml_file.read()
    deserialized = System.from_yaml(yaml_file)
    assert system == deserialized


def test_from_cif_builds_polymer_and_ligand_entities(cif_path, caplog):
    with caplog.at_level(logging.WARNING):
        system = System.from_cif(cif_path)

    protein = next(entity for entity in system.entities if isinstance(entity, Protein))
    ligand = next(entity for entity in system.entities if isinstance(entity, Ligand))
    assert protein.asym_ids == ["A"]
    assert protein.sequence == "ACDE"
    assert ligand.asym_ids == ["B", "C"]
    assert ligand.ccd_code == "EDO"
    assert system.asym_id_to_author_id == {"A": "A", "B": "B", "C": "C"}
    assert "Omitting CIF entity IDs" not in caplog.text


def test_from_cif_preserves_modified_residue_position(tmp_path: Path):
    cif_path = tmp_path / "modified.cif"
    cif_path.write_text(_MODIFIED_CIF)

    system = System.from_cif(cif_path)
    protein = next(entity for entity in system.entities if isinstance(entity, Protein))
    assert protein.sequence == "AXDE"
    assert protein.mods == [ModifiedResidue(res_id=2, ccd_code="SEP")]


def test_from_pdb_id_preserves_assembly_modifications(monkeypatch):
    response = Mock()
    response.content = gzip.compress(_ASSEMBLY_MODIFIED_CIF.encode())
    monkeypatch.setattr(
        "bioemu2.data_interface.system.requests.get",
        Mock(return_value=response),
    )

    system = System.from_pdb_id("3tkz")

    assert len(system.entities) == 1
    protein = system.entities[0]
    assert isinstance(protein, Protein)
    assert protein.asym_ids == ["B", "C"]
    assert protein.sequence == "RVIXFVPLNR"
    assert protein.mods == [ModifiedResidue(res_id=4, ccd_code="PTR")]
    assert system.asym_id_to_author_id == {"B": "P", "C": "Q"}


def test_from_cif_separates_chains_with_different_modifications(tmp_path: Path):
    cif_path = tmp_path / "assembly.cif"
    cif_path.write_text(_ASSEMBLY_MODIFIED_CIF.replace("C 4 PTR", "C 4 TYR"))

    system = System.from_cif(cif_path)

    assert len(system.entities) == 2
    modified, unmodified = system.entities
    assert isinstance(modified, Protein)
    assert modified.asym_ids == ["B"]
    assert modified.sequence == "RVIXFVPLNR"
    assert modified.mods == [ModifiedResidue(res_id=4, ccd_code="PTR")]
    assert isinstance(unmodified, Protein)
    assert unmodified.asym_ids == ["C"]
    assert unmodified.sequence == "RVIYFVPLNR"
    assert unmodified.mods == []


def test_from_cif_rejects_conflicting_modification_records(tmp_path: Path):
    cif_path = tmp_path / "conflicting.cif"
    cif_path.write_text(
        _ASSEMBLY_MODIFIED_CIF
        + """\
loop_
_pdbx_struct_mod_residue.label_asym_id
_pdbx_struct_mod_residue.label_comp_id
_pdbx_struct_mod_residue.label_seq_id
B SEP 4
#
"""
    )

    with pytest.raises(ValueError, match="Conflicting CIF monomers at chain B position 4"):
        System.from_cif(cif_path)


def test_from_cif_rejects_ambiguous_polymer_monomers(tmp_path: Path):
    cif_path = tmp_path / "ambiguous.cif"
    cif_path.write_text(_ASSEMBLY_MODIFIED_CIF.replace("2 4 PTR\n", "2 4 PTR\n2 4 SEP\n"))

    with pytest.raises(ValueError, match="Conflicting CIF monomers at entity 2 position 4"):
        System.from_cif(cif_path)


@pytest.mark.parametrize("with_raw_sequence", [False, True])
def test_from_cif_rejects_monomer_disagreeing_with_canonical_sequence(
    tmp_path: Path, with_raw_sequence: bool
):
    cif_path = tmp_path / "mismatched.cif"
    content = _ASSEMBLY_MODIFIED_CIF.replace("C 4 PTR", "C 4 ALA")
    if with_raw_sequence:
        content = content.replace(
            "_entity_poly.pdbx_seq_one_letter_code_can\n",
            "_entity_poly.pdbx_seq_one_letter_code_can\n_entity_poly.pdbx_seq_one_letter_code\n",
        ).replace(
            "2 'polypeptide(L)' RVIYFVPLNR\n",
            "2 'polypeptide(L)' RVIYFVPLNR 'RVI(PTR)FVPLNR'\n",
        )
    cif_path.write_text(content)

    with pytest.raises(ValueError, match="chain C position 4 disagrees with sequence residue TYR"):
        System.from_cif(cif_path)


def test_from_cif_leaves_unidentified_residues_unmodified(tmp_path: Path):
    cif_path = tmp_path / "unknown.cif"
    cif_path.write_text(
        _ASSEMBLY_MODIFIED_CIF.replace("RVIYFVPLNR", "RVIXFVPLNR").replace("PTR", "UNK")
    )

    system = System.from_cif(cif_path)

    assert system.entities[0].mods == []
    assert system.entities[0].sequence == "RVIXFVPLNR"


@pytest.mark.parametrize("with_annotations", [False, True])
def test_from_cif_aligns_expanded_modified_residue(tmp_path: Path, with_annotations: bool):
    cif_path = tmp_path / "chromophore.cif"
    content = _EXPANDED_MODIFIED_CIF
    if with_annotations:
        content += """\
loop_
_pdbx_struct_mod_residue.label_asym_id
_pdbx_struct_mod_residue.label_comp_id
_pdbx_struct_mod_residue.label_seq_id
A CRO 2
A CRO 2
A CRO 2
#
"""
    cif_path.write_text(content)

    system = System.from_cif(cif_path)

    protein = system.entities[0]
    assert isinstance(protein, Protein)
    assert protein.sequence == "AXD"
    assert protein.mods == [ModifiedResidue(res_id=2, ccd_code="CRO")]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            _EXPANDED_MODIFIED_CIF.replace("'A(CRO)D'", "'(ALA)(CRO)(ASP)'"),
            [
                Protein(
                    sequence="AXD", asym_ids=["A"], mods=[ModifiedResidue(res_id=2, ccd_code="CRO")]
                )
            ],
            id="parenthesized-standard-protein",
        ),
        pytest.param(
            _EXPANDED_MODIFIED_CIF.replace(
                """loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 CRO
1 3 ASP
#
""",
                "",
            ),
            [
                Protein(
                    sequence="AXD", asym_ids=["A"], mods=[ModifiedResidue(res_id=2, ccd_code="CRO")]
                )
            ],
            id="no-entity-monomer-table",
        ),
        pytest.param(
            _THYMINE_DIMER_CIF,
            [
                DNA(
                    sequence="ATCGGCNCGCGCA",
                    asym_ids=["C", "E"],
                    mods=[ModifiedResidue(res_id=7, ccd_code="WUH")],
                )
            ],
            id="8oy5-thymine-dimer",
        ),
    ],
)
def test_from_cif_decodes_raw_monomers(
    tmp_path: Path, content: str, expected: list[Protein | DNA]
) -> None:
    cif_path = tmp_path / "raw-monomers.cif"
    cif_path.write_text(content)

    system = System.from_cif(cif_path)

    assert system.entities == expected


def test_from_cif_recovers_modification_without_monomer_tables(tmp_path: Path) -> None:
    content = _EXPANDED_MODIFIED_CIF
    for category in ("_entity_poly_seq", "_pdbx_poly_seq_scheme"):
        start = content.index(f"loop_\n{category}.")
        end = content.index("#\n", start) + 2
        content = content[:start] + content[end:]
    cif_path = tmp_path / "raw-only.cif"
    cif_path.write_text(content)

    assert System.from_cif(cif_path).entities == [
        Protein(sequence="AXD", asym_ids=["A"], mods=[ModifiedResidue(res_id=2, ccd_code="CRO")])
    ]


def test_from_cif_partial_chain_scheme_uses_raw_positions(tmp_path: Path) -> None:
    content = _EXPANDED_MODIFIED_CIF
    start = content.index("loop_\n_entity_poly_seq.")
    end = content.index("#\n", start) + 2
    content = content[:start] + content[end:]
    content = content.replace("A 3 ASP\n", "")
    cif_path = tmp_path / "partial-chain-scheme.cif"
    cif_path.write_text(content)

    assert System.from_cif(cif_path).entities == [
        Protein(sequence="AXD", asym_ids=["A"], mods=[ModifiedResidue(res_id=2, ccd_code="CRO")])
    ]


def test_from_cif_preserves_chain_specific_standard_override(tmp_path: Path) -> None:
    content = _EXPANDED_MODIFIED_CIF.replace(
        "A 3 ASP\n#", "A 3 ASP\nB 1 ALA\nB 2 TYR\nB 3 ASP\n#"
    ).replace("A 1 A\n#", "A 1 A\nB 1 B\n#")
    cif_path = tmp_path / "override.cif"
    cif_path.write_text(content)

    assert System.from_cif(cif_path).entities == [
        Protein(sequence="AXD", asym_ids=["A"], mods=[ModifiedResidue(res_id=2, ccd_code="CRO")]),
        Protein(sequence="AYD", asym_ids=["B"]),
    ]


def test_from_cif_accepts_standard_override_matching_equal_length_canonical(tmp_path: Path) -> None:
    content = (
        _ASSEMBLY_MODIFIED_CIF.replace(
            "_entity_poly.pdbx_seq_one_letter_code_can\n",
            "_entity_poly.pdbx_seq_one_letter_code_can\n_entity_poly.pdbx_seq_one_letter_code\n",
        )
        .replace(
            "2 'polypeptide(L)' RVIYFVPLNR\n",
            "2 'polypeptide(L)' RVIYFVPLNR 'RVI(PTR)FVPLNR'\n",
        )
        .replace("C 4 PTR", "C 4 TYR")
    )
    cif_path = tmp_path / "matching-override.cif"
    cif_path.write_text(content)

    assert System.from_cif(cif_path).entities == [
        Protein(
            sequence="RVIXFVPLNR", asym_ids=["B"], mods=[ModifiedResidue(res_id=4, ccd_code="PTR")]
        ),
        Protein(sequence="RVIYFVPLNR", asym_ids=["C"]),
    ]


@pytest.mark.parametrize("raw", [".", "?"])
def test_from_cif_missing_raw_sequence_uses_canonical(tmp_path: Path, raw: str) -> None:
    content = _TEST_CIF.replace(
        "_entity_poly.pdbx_seq_one_letter_code_can\n",
        "_entity_poly.pdbx_seq_one_letter_code_can\n_entity_poly.pdbx_seq_one_letter_code\n",
    ).replace("'polypeptide(L)' ACDE\n", f"'polypeptide(L)' ACDE {raw}\n")
    cif_path = tmp_path / "missing-raw.cif"
    cif_path.write_text(content)

    assert System.from_cif(cif_path).entities[0].sequence == "ACDE"


@pytest.mark.parametrize(
    ("content", "error"),
    [
        pytest.param(
            _EXPANDED_MODIFIED_CIF.replace("'A(CRO)D'", "'A(SEP)D'"),
            "Conflicting CIF monomers at entity 1 position 2",
            id="raw-monomer-disagrees-with-entity",
        ),
        pytest.param(
            _EXPANDED_MODIFIED_CIF.replace("ATYGD", "AXE"),
            "raw sequence at position 3 disagrees with canonical residue E",
            id="raw-standard-disagrees-with-canonical",
        ),
        pytest.param(
            _EXPANDED_MODIFIED_CIF.replace("_entity_poly.pdbx_seq_one_letter_code\n", "")
            .replace(" 'A(CRO)D'", "")
            .replace(
                "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n1 1 ALA\n1 2 CRO\n1 3 ASP\n#\n",
                "",
            ),
            "Cannot align CIF entity 1 polymer positions with chain A",
            id="expanded-canonical-without-raw-or-entity-monomers",
        ),
    ],
)
def test_from_cif_rejects_conflicting_raw_sequences(
    tmp_path: Path, content: str, error: str
) -> None:
    cif_path = tmp_path / "conflicting-raw.cif"
    cif_path.write_text(content)

    with pytest.raises(ValueError, match=error):
        System.from_cif(cif_path)


def test_from_cif_excludes_selected_ligands(cif_path):
    system = System.from_cif(cif_path, exclude_ligands=["edo"])

    assert system.chain_ids == ["A"]
    assert system.asym_id_to_author_id == {"A": "A"}
    assert all(isinstance(entity, Protein) for entity in system.entities)


def test_from_cif_warns_about_unsupported_records(tmp_path, caplog):
    cif_path = tmp_path / "unsupported.cif"
    cif_path.write_text(_UNSUPPORTED_CIF)

    with caplog.at_level(logging.WARNING):
        system = System.from_cif(cif_path)

    assert system.chain_ids == ["A"]
    assert "Omitting CIF entity IDs [2]" in caplog.text
    assert "unsupported polymer type 'polysaccharide(D)'" in caplog.text
    assert "Ignoring _struct_conn records" in caplog.text


@pytest.mark.parametrize(
    ("content", "missing_category"),
    [
        (_TEST_CIF.replace("_entity_poly.", "_unused_entity_poly."), "_entity_poly"),
        (
            _TEST_CIF.replace("_pdbx_entity_nonpoly.", "_unused_entity_nonpoly."),
            "_pdbx_entity_nonpoly",
        ),
    ],
)
def test_from_cif_warns_about_missing_entity_metadata(tmp_path, caplog, content, missing_category):
    cif_path = tmp_path / "missing-metadata.cif"
    cif_path.write_text(content)

    with caplog.at_level(logging.WARNING):
        System.from_cif(cif_path)

    assert "no matching _entity_poly or _pdbx_entity_nonpoly record" in caplog.text
    assert f"absent categories: {missing_category}" in caplog.text


@pytest.mark.parametrize(
    ("assembly_id", "selector", "expected_url"),
    [
        pytest.param(
            "2",
            {"auth_asym_ids": ["A"]},
            "https://files.rcsb.org/download/7WOI-assembly2.cif.gz",
            id="biological-assembly-author-id",
        ),
        pytest.param(
            None,
            {"label_asym_ids": ["A"]},
            "https://files.rcsb.org/download/7WOI.cif",
            id="asymmetric-unit-label-id",
        ),
    ],
)
def test_from_pdb_id_downloads_cif(
    monkeypatch,
    cif_path,
    assembly_id,
    selector,
    expected_url,
):
    content = cif_path.read_bytes()
    response = Mock()
    response.content = gzip.compress(content) if assembly_id is not None else content
    get = Mock(return_value=response)
    monkeypatch.setattr("bioemu2.data_interface.system.requests.get", get)

    system = System.from_pdb_id(
        "7woi",
        assembly_id=assembly_id,
        include_ligands=False,
        **selector,
    )

    get.assert_called_once_with(expected_url, timeout=60)
    response.raise_for_status.assert_called_once_with()
    assert system.chain_ids == ["A"]
    assert system.asym_id_to_author_id == {"A": "A"}
    assert all(isinstance(entity, Protein) for entity in system.entities)


def test_from_pdb_id_selects_all_labels_for_author_id(monkeypatch):
    response = Mock()
    response.content = _AMBIGUOUS_AUTHOR_CIF.encode()
    monkeypatch.setattr(
        "bioemu2.data_interface.system.requests.get",
        Mock(return_value=response),
    )

    system = System.from_pdb_id("7woi", assembly_id=None, auth_asym_ids=["E"])

    assert system.chain_ids == ["A", "B"]
    assert system.asym_id_to_author_id == {"A": "E", "B": "E"}


def test_from_pdb_id_excludes_selected_ligands(monkeypatch, cif_path):
    response = Mock()
    response.content = cif_path.read_bytes()
    monkeypatch.setattr(
        "bioemu2.data_interface.system.requests.get",
        Mock(return_value=response),
    )

    system = System.from_pdb_id(
        "7woi",
        assembly_id=None,
        exclude_ligands=["EDO"],
    )

    assert system.chain_ids == ["A"]
    assert system.asym_id_to_author_id == {"A": "A"}


def test_rejects_ligand_exclusion_when_all_ligands_are_excluded(cif_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        System.from_cif(
            cif_path,
            include_ligands=False,
            exclude_ligands=["EDO"],
        )


def test_from_pdb_id_rejects_both_chain_id_types():
    with pytest.raises(ValueError, match="either auth_asym_ids or label_asym_ids"):
        System.from_pdb_id(
            "7woi",
            auth_asym_ids=["E"],
            label_asym_ids=["A"],
        )


def test_from_description_assigns_chain_ids():
    system = System.from_description(
        proteins=["MADCLTCEQ", "GGGG"],
        rna=["GCAU"],
        dna=["GCAT"],
        ligand_ccd_codes=["ATP"],
        ligand_smiles=["CC"],
    )
    assert system.chain_ids == ["A", "B", "C", "D", "E", "F"]
    assert [e.entity_type for e in system.entities] == [
        "protein",
        "protein",
        "rna",
        "dna",
        "ligand",
        "ligand",
    ]
    assert system.entities[4].ccd_code == "ATP"
    assert system.entities[5].smiles == "CC"


def test_from_description_merges_identical_sequences():
    system = System.from_description(proteins=["MADC", "MADC"])
    assert len(system.entities) == 1
    assert system.entities[0].asym_ids == ["A", "B"]


def test_from_description_roundtrip_yaml(tmp_path):
    system = System.from_description(proteins=["MADC"], ligand_ccd_codes=["ATP"])
    system.to_yaml(tmp_path / "s.yaml")
    assert System.from_yaml(tmp_path / "s.yaml") == system


def test_asym_ids_past_z():
    assert list(islice(_asym_ids(), 28))[-3:] == ["Z", "AA", "AB"]


def test_str_is_yaml(prot, ligand_ccd, tmp_path):
    system = System(entities=[prot, ligand_ccd])
    system.to_yaml(tmp_path / "s.yaml")
    assert str(system) == (tmp_path / "s.yaml").read_text()
