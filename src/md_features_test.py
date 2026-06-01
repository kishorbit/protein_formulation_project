import os
import numpy as np
import pandas as pd
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer
import mdtraj as md
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data/processed/md_trajectories", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

SIMULATION_STEPS = 500000   # 1ns (2fs timestep)
REPORT_INTERVAL  = 5000     # save frame every 10ps -> 100 frames
TEMPERATURE      = 300      # Kelvin
PLATFORM         = "CUDA"

def prepare_system(pdb_path):
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller   = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*unit.nanometer)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.HBonds
    )
    return modeller, system

def run_md(pdb_path, uid, traj_dir):
    traj_path = os.path.join(traj_dir, f"{uid}.dcd")
    top_path  = os.path.join(traj_dir, f"{uid}_top.pdb")

    if os.path.exists(traj_path) and os.path.exists(top_path):
        print(f"  {uid}: cached")
        return traj_path, top_path

    modeller, system = prepare_system(pdb_path)

    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE*unit.kelvin,
        1/unit.picosecond,
        0.002*unit.picoseconds
    )

    platform = mm.Platform.getPlatformByName(PLATFORM)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"    Minimizing energy...")
    simulation.minimizeEnergy()

    print(f"    Equilibrating...")
    simulation.step(5000)

    # Save topology
    with open(top_path, 'w') as f:
        app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(), f)

    # Production run
    print(f"    Production MD ({SIMULATION_STEPS:,} steps)...")
    simulation.reporters.append(app.DCDReporter(traj_path, REPORT_INTERVAL))
    simulation.reporters.append(app.StateDataReporter(
        f"logs/{uid}_md.log", REPORT_INTERVAL,
        step=True, potentialEnergy=True, temperature=True, progress=True,
        remainingTime=True, totalSteps=SIMULATION_STEPS))
    simulation.step(SIMULATION_STEPS)

    return traj_path, top_path

def extract_md_features(traj_path, top_path, uid):
    print(f"    Extracting features...")
    traj = md.load(traj_path, top=top_path)

    prot_idx  = traj.topology.select("protein")
    traj_prot = traj.atom_slice(prot_idx)

    rmsd = md.rmsd(traj_prot, traj_prot, 0) * 10   # nm -> Å
    rg   = md.compute_rg(traj_prot) * 10            # nm -> Å
    rmsf = md.rmsf(traj_prot, traj_prot, 0) * 10   # nm -> Å
    sasa = md.shrake_rupley(traj_prot, mode='residue')

    top = traj_prot.topology
    hydrophobic = ['ALA','VAL','ILE','LEU','MET','PHE','TRP','PRO','TYR']
    met_idx          = [r.index for r in top.residues if r.name == 'MET']
    trp_idx          = [r.index for r in top.residues if r.name == 'TRP']
    hydrophobic_idx  = [r.index for r in top.residues if r.name in hydrophobic]

    met_sasa_dyn  = sasa[:, met_idx].mean(axis=1)       if met_idx         else np.zeros(len(traj))
    trp_sasa_dyn  = sasa[:, trp_idx].mean(axis=1)       if trp_idx         else np.zeros(len(traj))
    hydro_sasa    = sasa[:, hydrophobic_idx].sum(axis=1) if hydrophobic_idx else np.zeros(len(traj))

    return {
        "protein_id":               uid,
        "md_rmsd_mean":             rmsd.mean(),
        "md_rmsd_max":              rmsd.max(),
        "md_rmsd_std":              rmsd.std(),
        "md_rg_mean":               rg.mean(),
        "md_rg_std":                rg.std(),
        "md_rg_max":                rg.max(),
        "md_rmsf_mean":             rmsf.mean(),
        "md_rmsf_max":              rmsf.max(),
        "md_rmsf_std":              rmsf.std(),
        "md_met_sasa_mean":         met_sasa_dyn.mean(),
        "md_met_sasa_std":          met_sasa_dyn.std(),
        "md_trp_sasa_mean":         trp_sasa_dyn.mean(),
        "md_trp_sasa_std":          trp_sasa_dyn.std(),
        "md_hydrophobic_sasa_mean": hydro_sasa.mean(),
        "md_hydrophobic_sasa_std":  hydro_sasa.std(),
        "md_unfolding_risk":        rmsd.max() / (rg.mean() + 1e-6),
        "md_flexibility":           rmsf.mean(),
    }

# ── Main ──────────────────────────────────────────────────
print("\nLoading protein list...")
prot_df = pd.read_csv("data/processed/protein_features.csv")
ids     = prot_df["protein_id"].unique()[:5]   # TEST: first 5 only
pdb_dir  = "data/processed/alphafold_pdbs"
traj_dir = "data/processed/md_trajectories"

print(f"  Proteins: {list(ids)}")
print(f"  Steps: {SIMULATION_STEPS:,} (~1ns each)")
print(f"  Platform: {PLATFORM}\n")

os.makedirs("logs", exist_ok=True)

all_features, failed = [], []

for i, uid in enumerate(ids):
    pdb_path = os.path.join(pdb_dir, f"{uid}.pdb")
    if not os.path.exists(pdb_path):
        print(f"[{i+1}/5] {uid}: no PDB, skipping")
        failed.append(uid)
        continue

    print(f"\n[{i+1}/5] {uid}")
    try:
        traj_path, top_path = run_md(pdb_path, uid, traj_dir)
        feats = extract_md_features(traj_path, top_path, uid)
        all_features.append(feats)
        print(f"    OK — RMSD={feats['md_rmsd_mean']:.2f}A  "
              f"Rg={feats['md_rg_mean']:.2f}A  "
              f"flex={feats['md_flexibility']:.3f}  "
              f"hydro_sasa={feats['md_hydrophobic_sasa_mean']:.3f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        failed.append(uid)

md_df = pd.DataFrame(all_features)
md_df.to_csv("outputs/reports/md_features_test.csv", index=False)

print(f"\n{'='*55}")
print("MD TEST COMPLETE")
print("="*55)
print(f"  Processed: {len(all_features)}/5")
print(f"  Failed:    {len(failed)}")
print(f"\n  Feature summary:")
for col in ["md_rmsd_mean","md_rg_mean","md_rmsf_mean",
            "md_hydrophobic_sasa_mean","md_unfolding_risk"]:
    if col in md_df.columns:
        print(f"    {col:<28} mean={md_df[col].mean():.3f}  std={md_df[col].std():.3f}")
print(f"\nSaved: outputs/reports/md_features_test.csv")
print("="*55)
