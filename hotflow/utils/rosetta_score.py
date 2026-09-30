"""Rosetta FastRelax + InterfaceAnalyzer scoring.

Mirrors the post-generation scoring protocol used in PepFlow and PepHAR
(`PepFlowww/eval/energy.py`, `PepHAR/eval/energy.py`):

  for i in range(N):
      FastRelax(pose)
      stab_i = scorefxn(pose)
      InterfaceAnalyzerMover(pose)
      bind_i = pose.scores['dG_separated']

  return mean(stab), mean(bind)

PepFlow uses N=5 iterations; PepHAR uses N=2. We default to 2 (faster) and
expose it as a parameter.

pyrosetta is imported lazily so this file can be imported on machines without
pyrosetta installed — the import error is only raised when scoring is actually
attempted.
"""

from __future__ import annotations

from typing import List, Optional

_PYROSETTA_INITIALIZED = False


def _import_pyrosetta():
    """Lazy import. Raises ImportError with an install hint if unavailable."""
    try:
        import pyrosetta  # noqa: F401
        from pyrosetta import get_fa_scorefxn, init  # noqa: F401
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover  # noqa: F401
        from pyrosetta.rosetta.protocols.relax import FastRelax  # noqa: F401

        return pyrosetta, init, get_fa_scorefxn, FastRelax, InterfaceAnalyzerMover
    except ImportError as e:
        raise ImportError(
            "pyrosetta is required for Rosetta FastRelax scoring but is not "
            "installed in this environment. Install it from "
            "https://www.pyrosetta.org/downloads, or omit the --rosetta flag."
        ) from e


def init_pyrosetta(silent: bool = True) -> None:
    """Initialize pyrosetta exactly once per process."""
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return
    _, init, *_ = _import_pyrosetta()
    if silent:
        init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false")
    else:
        init()
    _PYROSETTA_INITIALIZED = True


def _chain_ids_from_pdb(pdb_path: str) -> List[str]:
    """Return chain IDs from a PDB that contain at least one standard amino
    acid residue with a CA atom, in file order."""
    from Bio.PDB import PDBParser, is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    chains: List[str] = []
    for model in structure:
        for chain in model:
            if any(is_aa(res) and res.has_id("CA") for res in chain):
                if chain.id not in chains:
                    chains.append(chain.id)
        break  # first model only
    return chains


def fast_relax_score(
    pdb_path: str,
    peptide_chain: str,
    num_iterations: int = 2,
    silent: bool = True,
    out_pdb: Optional[str] = None,
) -> dict:
    """FastRelax + InterfaceAnalyzer scoring of a peptide-receptor complex.

    Args:
        pdb_path: path to PDB file containing peptide + receptor.
        peptide_chain: chain ID of the (generated) peptide. The interface for
            InterfaceAnalyzerMover is built as `<peptide_chain>_<rest>`.
        num_iterations: number of FastRelax + score repeats. PepHAR uses 2,
            PepFlow uses 5. Defaults to 2.
        silent: suppress pyrosetta stdout.
        out_pdb: if given, dump the relaxed pose to this path after the final
            iteration. Downstream metric computation should read this file
            (post-relax structure) rather than `pdb_path` (raw generation).

    Returns:
        dict with keys:
          success (bool)
          stab (float | None)   — mean full-atom REU after FastRelax
          bind (float | None)   — mean dG_separated (kcal/mol)
          n_iter (int)
          interface (str | None)
          relaxed_pdb (str | None)  — path to the dumped relaxed PDB, or None
          error (str | None)
    """
    pdb_path = str(pdb_path)
    try:
        pyrosetta, _, get_fa_scorefxn, FastRelax, InterfaceAnalyzerMover = _import_pyrosetta()
        init_pyrosetta(silent=silent)

        all_chains = _chain_ids_from_pdb(pdb_path)
        if peptide_chain not in all_chains:
            return {
                "success": False,
                "stab": None,
                "bind": None,
                "n_iter": 0,
                "interface": None,
                "relaxed_pdb": None,
                "error": f"peptide chain '{peptide_chain}' not in PDB chains {all_chains}",
            }
        receptor_chains = [c for c in all_chains if c != peptide_chain]
        if not receptor_chains:
            return {
                "success": False,
                "stab": None,
                "bind": None,
                "n_iter": 0,
                "interface": None,
                "relaxed_pdb": None,
                "error": "no receptor chain found in PDB",
            }
        interface = f"{peptide_chain}_{''.join(receptor_chains)}"

        pose = pyrosetta.pose_from_pdb(pdb_path)
        scorefxn = get_fa_scorefxn()
        fast_relax = FastRelax()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)

        stabs: List[float] = []
        binds: List[float] = []
        for _ in range(num_iterations):
            fast_relax.apply(pose)
            stabs.append(float(scorefxn(pose)))
            mover.apply(pose)
            binds.append(float(pose.scores["dG_separated"]))

        relaxed_path: Optional[str] = None
        if out_pdb is not None:
            relaxed_path = str(out_pdb)
            pose.dump_pdb(relaxed_path)

        return {
            "success": True,
            "stab": sum(stabs) / len(stabs),
            "bind": sum(binds) / len(binds),
            "n_iter": num_iterations,
            "interface": interface,
            "relaxed_pdb": relaxed_path,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "stab": None,
            "bind": None,
            "n_iter": 0,
            "interface": None,
            "relaxed_pdb": None,
            "error": f"{type(e).__name__}: {e}",
        }


def score_only(
    pdb_path: str,
    peptide_chain: str,
    silent: bool = True,
) -> dict:
    """Score a PDB without FastRelax — raw structure energy.

    Same as fast_relax_score but skips the FastRelax step entirely.
    Returns the energy of the structure as-is.
    """
    pdb_path = str(pdb_path)
    try:
        pyrosetta, _, get_fa_scorefxn, _FastRelax, InterfaceAnalyzerMover = _import_pyrosetta()
        init_pyrosetta(silent=silent)

        all_chains = _chain_ids_from_pdb(pdb_path)
        if peptide_chain not in all_chains:
            return {
                "success": False, "stab": None, "bind": None,
                "interface": None,
                "error": f"peptide chain '{peptide_chain}' not in PDB chains {all_chains}",
            }
        receptor_chains = [c for c in all_chains if c != peptide_chain]
        if not receptor_chains:
            return {
                "success": False, "stab": None, "bind": None,
                "interface": None, "error": "no receptor chain found in PDB",
            }
        interface = f"{peptide_chain}_{''.join(receptor_chains)}"

        pose = pyrosetta.pose_from_pdb(pdb_path)
        scorefxn = get_fa_scorefxn()

        stab = float(scorefxn(pose))
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        mover.apply(pose)
        bind = float(pose.scores["dG_separated"])

        return {
            "success": True, "stab": stab, "bind": bind,
            "interface": interface, "error": None,
        }
    except Exception as e:
        return {
            "success": False, "stab": None, "bind": None,
            "interface": None, "error": f"{type(e).__name__}: {e}",
        }
