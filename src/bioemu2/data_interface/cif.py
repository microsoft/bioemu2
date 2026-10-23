"""Utilities for constructing system entities from mmCIF files."""

import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from biotite.structure.io import pdbx
from pdbecif.mmcif_tools import MMCIF2Dict

from .const import PROT_LETTER_TO_TOKEN, Token

logger = logging.getLogger(__name__)

_UNKNOWN_MONOMERS = {"protein": "UNK", "rna": "N", "dna": "DN"}
_STANDARD_MONOMER_TO_LETTER = {
    "protein": {
        token.name: letter
        for letter, token in PROT_LETTER_TO_TOKEN.items()
        if token not in (Token.UNK_PROT, Token.GAP)
    },
    "rna": {letter: letter for letter in "ACGU"},
    "dna": {f"D{letter}": letter for letter in "ACGT"},
}


def _cif_rows(table: dict[str, Any], *columns: str) -> Iterator[tuple[str, ...]]:
    values = (
        [table[column]] if isinstance(table[column], str) else table[column] for column in columns
    )
    yield from zip(*values, strict=True)


def _add_monomer(positions: dict[int, str], position: int, ccd_code: str, source: str) -> None:
    previous = positions.setdefault(position, ccd_code)
    if previous != ccd_code:
        raise ValueError(
            f"Conflicting CIF monomers at {source} position {position}: {previous} and {ccd_code}."
        )


def _positioned_sequence(
    sequence: str,
    raw_sequence: str | None,
    monomers: dict[int, str],
    entity_type: str,
    unknown_residue: str,
    entity_id: int,
) -> tuple[str, dict[int, str]]:
    """Position raw polymer tokens and retain parenthesized monomer identities."""
    if raw_sequence is None:
        if monomers and (
            len(sequence) != len(monomers) or set(monomers) != set(range(1, len(sequence) + 1))
        ):
            raise ValueError(
                f"CIF entity {entity_id} has {len(monomers)} polymer positions but "
                f"{len(sequence)} canonical sequence letters and no unexpanded sequence."
            )
        return sequence, {}
    raw_sequence = "".join(raw_sequence.split())
    tokens = re.findall(r"\([^()]+\)|[^()]", raw_sequence)
    if "".join(tokens) != raw_sequence or (
        monomers and set(monomers) != set(range(1, len(tokens) + 1))
    ):
        raise ValueError(
            f"Cannot align CIF entity {entity_id} polymer positions with its sequence."
        )
    raw_monomers: dict[int, str] = {}
    letters: list[str] = []
    for position, token in enumerate(tokens, start=1):
        if token.startswith("("):
            ccd_code = token[1:-1]
            _add_monomer(raw_monomers, position, ccd_code, f"entity {entity_id}")
            letters.append(_STANDARD_MONOMER_TO_LETTER[entity_type].get(ccd_code, unknown_residue))
        else:
            letters.append(token)
        if (
            len(sequence) == len(tokens)
            and sequence[position - 1] != unknown_residue
            and letters[-1] != unknown_residue
            and sequence[position - 1] != letters[-1]
        ):
            raise ValueError(
                f"CIF entity {entity_id} raw sequence at position {position} "
                f"disagrees with canonical residue {sequence[position - 1]}."
            )
    return "".join(letters), raw_monomers


def author_chain_ids_from_cif_file(cif_file: str | Path) -> dict[str, str]:
    """Map label chain IDs to author chain IDs."""
    cif = pdbx.CIFFile.read(str(cif_file))
    block = cif[next(iter(cif))]
    atom_site = block["atom_site"]
    label_asym_ids = atom_site["label_asym_id"].as_array(str)
    auth_asym_ids = atom_site["auth_asym_id"].as_array(str)

    mapping: dict[str, str] = {}
    for label_asym_id, auth_asym_id in zip(label_asym_ids, auth_asym_ids, strict=True):
        previous = mapping.setdefault(label_asym_id, auth_asym_id)
        if previous != auth_asym_id:
            raise ValueError(
                f"Label chain '{label_asym_id}' maps to multiple author chain IDs: "
                f"'{previous}' and '{auth_asym_id}'."
            )
    return mapping


def schema_from_cif_file(cif_file: str | Path) -> list[dict[str, dict[str, Any]]]:
    """Extract a molecular-system schema from a local mmCIF file.

    Supports protein, DNA, RNA, small-molecule ligand chains, and modified
    polymer residues. Polymer sequence records identify modified monomers even
    when the file does not contain explicit modification annotations.

    Branched polymers (e.g. glycans), covalently bound ligands, and simplified mmCIF files
    containing only an _atom_site table are not supported. Common non-biological molecules (e.g.
    crystallization additives and detergents) are not automatically excluded.
    """
    parser = MMCIF2Dict()
    parsed_data = parser.parse(str(cif_file))
    assert len(parsed_data) == 1, "More than one data block present in CIF file"
    cif_data = parsed_data[next(iter(parsed_data))]

    # Make a map from entity ID to label chain IDs.
    entity_id_to_label_asym_ids: dict[int, list[str]] = {}
    atom_site_table = cif_data["_atom_site"]
    label_asym_ids = atom_site_table["label_asym_id"]
    entity_ids = atom_site_table["label_entity_id"]
    if isinstance(label_asym_ids, str):
        label_asym_ids = [label_asym_ids]
        entity_ids = [entity_ids]
    for label_asym_id, entity_id in zip(label_asym_ids, entity_ids, strict=True):
        entity_label_asym_ids = entity_id_to_label_asym_ids.setdefault(int(entity_id), [])
        if label_asym_id not in entity_label_asym_ids:
            entity_label_asym_ids.append(label_asym_id)

    # Check which entity ids are protein/rna/dna
    entity_id_to_poly_type: dict[int, str] = {}
    entity_id_to_poly_sequence: dict[int, str] = {}
    entity_id_to_raw_sequence: dict[int, str] = {}
    if "_entity_poly" in cif_data:
        entity_poly_table = cif_data["_entity_poly"]
        if isinstance(entity_poly_table["entity_id"], str):
            # single entry, not a _loop table
            entity_id_to_poly_type[int(entity_poly_table["entity_id"])] = entity_poly_table["type"]
            entity_id_to_poly_sequence[int(entity_poly_table["entity_id"])] = "".join(
                entity_poly_table["pdbx_seq_one_letter_code_can"].splitlines()
            )
            if "pdbx_seq_one_letter_code" in entity_poly_table:
                raw_sequence = entity_poly_table["pdbx_seq_one_letter_code"]
                if raw_sequence not in (".", "?"):
                    entity_id_to_raw_sequence[int(entity_poly_table["entity_id"])] = raw_sequence
        else:
            for entity_id, entity_type, entity_sequence in _cif_rows(
                entity_poly_table, "entity_id", "type", "pdbx_seq_one_letter_code_can"
            ):
                entity_id_to_poly_type[int(entity_id)] = entity_type
                entity_id_to_poly_sequence[int(entity_id)] = "".join(entity_sequence.splitlines())
            if "pdbx_seq_one_letter_code" in entity_poly_table:
                for entity_id, raw_sequence in _cif_rows(
                    entity_poly_table, "entity_id", "pdbx_seq_one_letter_code"
                ):
                    if raw_sequence not in (".", "?"):
                        entity_id_to_raw_sequence[int(entity_id)] = raw_sequence
    # Check which entity ids are non-polymers (ligands)
    entity_id_to_ccd_code: dict[int, str] = {}
    if "_pdbx_entity_nonpoly" in cif_data:
        entity_nonpoly_table = cif_data["_pdbx_entity_nonpoly"]
        if isinstance(entity_nonpoly_table["entity_id"], str):
            # single entry, not a _loop table
            entity_id_to_ccd_code[int(entity_nonpoly_table["entity_id"])] = entity_nonpoly_table[
                "comp_id"
            ]
        else:
            for entity_id, ccd_code in zip(
                entity_nonpoly_table["entity_id"],
                entity_nonpoly_table["comp_id"],
                strict=True,
            ):
                entity_id_to_ccd_code[int(entity_id)] = ccd_code

    supported_poly_types = {
        "polypeptide(L)": ("protein", "X"),
        "polyribonucleotide": ("rna", "N"),
        "polydeoxyribonucleotide": ("dna", "N"),
    }
    supported_entity_ids = {
        entity_id
        for entity_id, poly_type in entity_id_to_poly_type.items()
        if poly_type in supported_poly_types
    } | entity_id_to_ccd_code.keys()
    unsupported_entity_ids = sorted(entity_id_to_label_asym_ids.keys() - supported_entity_ids)
    if unsupported_entity_ids:
        missing_categories = [
            category
            for category in ("_entity_poly", "_pdbx_entity_nonpoly")
            if category not in cif_data
        ]
        reasons = {
            entity_id: (
                f"unsupported polymer type {entity_id_to_poly_type[entity_id]!r}"
                if entity_id in entity_id_to_poly_type
                else "no matching _entity_poly or _pdbx_entity_nonpoly record"
                + (
                    f"; absent categories: {', '.join(missing_categories)}"
                    if missing_categories
                    else ""
                )
            )
            for entity_id in unsupported_entity_ids
        }
        logger.warning(
            "Omitting CIF entity IDs %s: %s.",
            unsupported_entity_ids,
            reasons,
        )

    if "_struct_conn" in cif_data:
        logger.warning(
            "Ignoring _struct_conn records in CIF file %s. Explicit inter-residue and inter-chain "
            "bonds are not imported.",
            cif_file,
        )

    entity_monomers: dict[int, dict[int, str]] = defaultdict(dict)
    if "_entity_poly_seq" in cif_data:
        for entity_id, seq_id, ccd_code in _cif_rows(
            cif_data["_entity_poly_seq"], "entity_id", "num", "mon_id"
        ):
            _add_monomer(
                entity_monomers[int(entity_id)], int(seq_id), ccd_code, f"entity {entity_id}"
            )

    chain_monomers: dict[str, dict[int, str]] = defaultdict(dict)
    if "_pdbx_poly_seq_scheme" in cif_data:
        for asym_id, seq_id, ccd_code in _cif_rows(
            cif_data["_pdbx_poly_seq_scheme"], "asym_id", "seq_id", "mon_id"
        ):
            _add_monomer(chain_monomers[asym_id], int(seq_id), ccd_code, f"chain {asym_id}")

    label_asym_id_to_mods: dict[str, dict[int, str]] = defaultdict(dict)
    if "_pdbx_struct_mod_residue" in cif_data:
        for asym_id, ccd_code, seq_id in _cif_rows(
            cif_data["_pdbx_struct_mod_residue"], "label_asym_id", "label_comp_id", "label_seq_id"
        ):
            _add_monomer(label_asym_id_to_mods[asym_id], int(seq_id), ccd_code, f"chain {asym_id}")

    schema: list[dict[str, dict[str, Any]]] = []
    # Create sampling schema
    entity_ids = sorted(supported_entity_ids)
    for entity_id in entity_ids:
        if entity_id in entity_id_to_poly_type:
            poly_type = entity_id_to_poly_type[entity_id]
            canonical_sequence = entity_id_to_poly_sequence[entity_id]
            poly_asym_ids = entity_id_to_label_asym_ids[entity_id]

            if poly_type == "polypeptide(L)":  # protein
                entity_type = "protein"
                unknown_residue = "X"
            elif poly_type == "polyribonucleotide":  # rna
                entity_type = "rna"
                unknown_residue = "N"
            elif poly_type == "polydeoxyribonucleotide":  # dna
                entity_type = "dna"
                unknown_residue = "N"
            else:
                raise ValueError(f"Not recognised polymer type: {poly_type}")

            poly_sequence, raw_monomers = _positioned_sequence(
                canonical_sequence,
                entity_id_to_raw_sequence.get(entity_id),
                entity_monomers.get(entity_id, {}),
                entity_type,
                unknown_residue,
                entity_id,
            )
            if not entity_monomers[entity_id] and entity_id not in entity_id_to_raw_sequence:
                for asym_id in poly_asym_ids:
                    if not chain_monomers[asym_id]:
                        continue
                    positions = chain_monomers[asym_id]
                    if set(positions) == set(range(1, max(positions) + 1)):
                        if len(poly_sequence) != len(positions):
                            raise ValueError(
                                f"Cannot align CIF entity {entity_id} polymer positions "
                                f"with chain {asym_id}."
                            )
            for position, ccd_code in raw_monomers.items():
                _add_monomer(entity_monomers[entity_id], position, ccd_code, f"entity {entity_id}")
            grouped_chains: dict[tuple[str, tuple[tuple[int, str], ...]], list[str]] = {}
            for asym_id in poly_asym_ids:
                monomers = {**entity_monomers.get(entity_id, {}), **chain_monomers.get(asym_id, {})}
                annotations = label_asym_id_to_mods.get(asym_id, {})
                mods: dict[int, str] = {}
                for position, ccd_code in monomers.items():
                    if not 1 <= position <= len(poly_sequence) or ccd_code in (".", "?"):
                        raise ValueError(
                            f"Invalid CIF monomer {ccd_code} at chain {asym_id} position {position}."
                        )
                    if ccd_code in _STANDARD_MONOMER_TO_LETTER[entity_type]:
                        sequence_letter = poly_sequence[position - 1]
                        if (
                            sequence_letter == unknown_residue
                            and position in raw_monomers
                            and len(canonical_sequence) == len(poly_sequence)
                        ):
                            sequence_letter = canonical_sequence[position - 1]
                        if sequence_letter != unknown_residue:
                            expected = (
                                PROT_LETTER_TO_TOKEN[sequence_letter].name
                                if entity_type == "protein"
                                else f"D{sequence_letter}"
                                if entity_type == "dna"
                                else sequence_letter
                            )
                            if ccd_code != expected:
                                raise ValueError(
                                    f"CIF monomer {ccd_code} at chain {asym_id} position "
                                    f"{position} disagrees with sequence residue {expected}."
                                )
                    elif ccd_code != _UNKNOWN_MONOMERS[entity_type]:
                        mods[position] = ccd_code
                for position, ccd_code in annotations.items():
                    if not 1 <= position <= len(poly_sequence):
                        raise ValueError(
                            f"Invalid CIF modification at chain {asym_id} position {position}."
                        )
                    if (
                        position in chain_monomers.get(asym_id, {})
                        and chain_monomers[asym_id][position] != ccd_code
                    ):
                        raise ValueError(
                            f"Conflicting CIF monomers at chain {asym_id} position {position}: "
                            f"{chain_monomers[asym_id][position]} and {ccd_code}."
                        )
                    if position in mods and mods[position] != ccd_code:
                        raise ValueError(
                            f"Conflicting CIF monomers at chain {asym_id} position {position}: "
                            f"{mods[position]} and {ccd_code}."
                        )
                    mods[position] = ccd_code
                chain_sequence = list(poly_sequence)
                for position, ccd_code in chain_monomers[asym_id].items():
                    if (
                        position not in mods
                        and ccd_code in _STANDARD_MONOMER_TO_LETTER[entity_type]
                    ):
                        chain_sequence[position - 1] = _STANDARD_MONOMER_TO_LETTER[entity_type][
                            ccd_code
                        ]
                for position in mods:
                    chain_sequence[position - 1] = unknown_residue
                grouped_chains.setdefault(
                    ("".join(chain_sequence), tuple(sorted(mods.items()))), []
                ).append(asym_id)

            for (positioned_chain_sequence, grouped_mods), asym_ids in grouped_chains.items():
                schema.append(
                    {
                        entity_type: {
                            "id": asym_ids,
                            "sequence": positioned_chain_sequence,
                            "mods": [
                                {"res_id": position, "ccd_code": ccd_code}
                                for position, ccd_code in grouped_mods
                            ],
                        }
                    }
                )
        else:
            ccd_code = entity_id_to_ccd_code[entity_id]
            if ccd_code != "HOH":  # discard waters
                schema.append(
                    {
                        "ligand": {
                            "id": entity_id_to_label_asym_ids[entity_id],
                            "ccd": ccd_code,
                        }
                    }
                )

    logger.info(
        "Extracted schema from CIF file %s:\n```\n%s```",
        cif_file,
        yaml.dump(schema, indent=2),
    )
    return schema


def system_entities_from_cif_file(cif_file: str | Path) -> list[dict[str, Any]]:
    """Extract System entity dictionaries from a local mmCIF file. See `schema_from_cif_file`
    for details.
    """
    entities: list[dict[str, Any]] = []
    for item in schema_from_cif_file(cif_file):
        entity_type, spec = next(iter(item.items()))
        entity: dict[str, Any] = {
            "entity_type": entity_type,
            "asym_ids": spec["id"],
        }
        if entity_type == "ligand":
            entity["ccd_code"] = spec["ccd"]
        else:
            entity["sequence"] = spec["sequence"]
            entity["mods"] = spec.get("mods", [])
        entities.append(entity)
    return entities
