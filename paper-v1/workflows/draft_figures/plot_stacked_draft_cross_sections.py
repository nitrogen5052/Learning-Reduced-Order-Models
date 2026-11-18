"""Single-panel stacked draft posterior-predictive cross sections."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
SOURCE=ROOT/"Scratch/CodeRestructure/global_diagnostics/curated_neutron_map"
DATA=HERE/"data/draft_model_error_posterior"
OUT=HERE/"figures/paper_results_drafts"; OUT.mkdir(parents=True,exist_ok=True)
CASES=[
    (40,20,"Ca",13.90,8,"#2864B7"),
    (54,26,"Fe",16.93,6,"#2A9D8F"),
    (60,28,"Ni",13.94,4,"#7A5195"),
    (120,50,"Sn",13.92,2,"#E09F3E"),
    (208,82,"Pb",16.90,0,"#C44536"),
]

archive=np.load(SOURCE/"curated_kduq_sampling_payload.npz")
predictive=np.load(DATA/"posterior_predictive.npz")
parameter=predictive["parameter_predictions"]
total=predictive["total_predictions"]
model_fraction=float(predictive["model_fraction"])
points=pd.read_csv(SOURCE/"map_fit_points.csv")
verified=np.load(SOURCE/"refined_fom_verification.npz")
canonical=verified["canonical"]
case_a,case_z,case_e=archive["case_a"].astype(int),archive["case_z"].astype(int),archive["case_energy"]
obs_case=archive["observation_case"].astype(int); observed=archive["observed"]
angles=points.angle_cm_deg.to_numpy(float); errors=points.point_error_mb_sr.to_numpy(float)

q025,q16,q84,q975=np.quantile(total,[.025,.16,.84,.975],axis=0)
median=np.median(parameter,axis=0)
fig,ax=plt.subplots(figsize=(6.6,8.2),layout="constrained")
for a,z,symbol,wanted,exponent,color in CASES:
    candidates=np.flatnonzero((case_a==a)&(case_z==z))
    case=candidates[np.argmin(abs(case_e[candidates]-wanted))]
    idx=np.flatnonzero(obs_case==case); idx=idx[np.argsort(angles[idx])]
    theta=angles[idx]; scale=10.**exponent
    ax.fill_between(theta,scale*q025[idx],scale*q975[idx],color=color,alpha=.18,lw=0)
    ax.fill_between(theta,scale*q16[idx],scale*q84[idx],color=color,alpha=.36,lw=0)
    ax.plot(theta,scale*median[idx],color=color,lw=2.0)
    ax.plot(theta,scale*canonical[idx],color=".20",lw=1.15,ls="--")
    ax.errorbar(theta,scale*observed[idx],yerr=scale*errors[idx],fmt="o",ms=3.4,
                mfc="black",mec="black",ecolor=".35",elinewidth=.65,capsize=0,zorder=5)
    # Anchor annotations just above the local 95% band at each curve end.
    # This keeps every label attached to its distribution without covering it.
    left_height = scale*q975[idx][0]*1.75
    right_height = scale*q975[idx][-1]*1.75
    ax.text(2.0, left_height, rf"$\times 10^{{{exponent}}}$",
            color=color, fontsize=13, ha="left", va="center")
    ax.text(177.0, right_height,
            rf"$^{{{a}}}\mathrm{{{symbol}}}$, {case_e[case]:g} MeV",
            color=color, fontsize=13, ha="right", va="center")
ax.set_yscale("log"); ax.set_xlim(0,180)
ax.set_xlabel(r"$\theta_{\rm cm}$ [deg]", fontsize=16)
ax.set_ylabel(r"scaled $d\sigma/d\Omega$ [mb/sr]", fontsize=16)
ax.tick_params(axis="both", which="both", labelsize=13)
ax.grid(False)
handles=[
    Line2D([],[],color=".20",ls="--",lw=1.4,label="canonical KD (FOM)"),
    Line2D([],[],color=".30",lw=2,label="posterior median (LROM)"),
    Line2D([],[],marker="o",ls="none",color="black",markersize=4,label="curated data"),
    Patch(facecolor=".35",alpha=.36,label="68% posterior predictive"),
    Patch(facecolor=".35",alpha=.18,label="95% posterior predictive"),
]
ax.legend(handles=handles,loc="lower left",frameon=True,ncol=1)
for ext in ("png","pdf"):
    fig.savefig(OUT/f"figure_04_kduq_stacked_model_error_draft.{ext}",dpi=300,bbox_inches="tight")
print(OUT/"figure_04_kduq_stacked_model_error_draft.png")
