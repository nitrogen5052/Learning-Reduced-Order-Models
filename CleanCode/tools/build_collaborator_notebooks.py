"""Upgrade notebooks 01/02 and create the detailed real-data notebook 03."""

from __future__ import annotations

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def clear(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}


def enhance_01():
    path = NOTEBOOKS / "01_single_wavefunction.ipynb"
    nb = nbf.read(path, as_version=4)
    if any("RIDGE_AND_BACKEND_EXTENSION" in c.source for c in nb.cells):
        return
    setup = nb.cells[2].source
    setup = setup.replace(
        "import numpy as np",
        "import os, sys, time\nfrom pathlib import Path\n"
        "ROOT = Path.cwd().resolve()\n"
        "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib-cache'))\n\n"
        "import numpy as np",
    )
    setup = setup.replace(
        "from lrom import BasisConfig, MaxVol, ScatteringLROM, ScatteringProblem",
        "from lrom import (BasisConfig, MaxVol, ScatteringLROM, "
        "ScatteringProblem, get_runtime)",
    )
    setup += """

BACKEND = os.getenv("LROM_BACKEND", "cpu")
PRECISION = os.getenv("LROM_PRECISION", "double")
runtime = get_runtime(BACKEND, PRECISION)
print(f"online runtime: {runtime.name} on {runtime.device} ({runtime.precision})")
"""
    nb.cells[2].source = setup
    extension = [
        md(r"""
        <!-- RIDGE_AND_BACKEND_EXTENSION -->
        # 3. Stability of the learned implicit equation

        The preceding sections reproduce the original detailed wavefunction
        study. We now add the lessons from the stability investigation without
        changing the physics, snapshots, reduced bases, or predictor locations.

        OLS minimizes the stacked implicit residual. Ridge instead minimizes

        $$\|D\theta-t\|_2^2+\alpha\|\theta\|_2^2,\qquad
        \alpha=\lambda\sigma_{\max}(D)^2.$$

        Thus `ridge=1e-14` is relative to the scale of the design matrix. It
        suppresses weakly identified, mutually cancelling coefficient
        directions while leaving well-identified directions almost unchanged.
        """),
        code(r"""
        RIDGE = 1e-14

        def implicit_design(learned, data, channel=(0, 0.0)):
            coordinates = learned.basis.project(data.wavefunctions[channel])
            features = learned.features(data.potentials[channel])
            blocks = np.concatenate([
                np.einsum("sf,si->sfi", features, coordinates, optimize=True),
                -features[..., None],
            ], axis=2)
            return blocks.reshape(len(features), features.shape[1]*(coordinates.shape[1]+1))

        ridge_vv = ScatteringLROM(
            problem, BasisConfig(size=BASIS_SIZE), MaxVol(count=1, ridge=RIDGE)
        ).train(training_data_vv, precomputed_bases={(0,0.0): model_vv.basis})
        ridge_vv_wave = np.asarray([
            ridge_vv.predict(row).wavefunction((0,0.0)) for row in test_samples_vv
        ])
        ols_vv_wave = lrom_test_vv
        ols_vv_condition = np.asarray([
            emulator_vv.predict(row).condition_number((0,0.0)) for row in test_samples_vv
        ])
        ridge_vv_condition = np.asarray([
            ridge_vv.predict(row).condition_number((0,0.0)) for row in test_samples_vv
        ])

        design_vv = implicit_design(model_vv, training_data_vv)
        singular_vv = np.linalg.svd(design_vv, compute_uv=False)
        pd.DataFrame({
            "fit": ["OLS", "ridge-14"],
            "median test relative L2": [
                np.median(relative_l2(ols_vv_wave, test_phi_vv)),
                np.median(relative_l2(ridge_vv_wave, test_phi_vv)),
            ],
            "maximum test relative L2": [
                np.max(relative_l2(ols_vv_wave, test_phi_vv)),
                np.max(relative_l2(ridge_vv_wave, test_phi_vv)),
            ],
            "maximum online condition": [ols_vv_condition.max(), ridge_vv_condition.max()],
        })
        """),
        code(r"""
        fig, axes = plt.subplots(1, 3, figsize=(15,4), layout="constrained")
        axes[0].semilogy(singular_vv/singular_vv[0], "o-")
        axes[0].axhline(np.sqrt(RIDGE), color="tab:red", ls="--", label=r"$\sqrt{\lambda}$")
        axes[0].set(xlabel="singular-value index", ylabel=r"$\sigma_i(D)/\sigma_1(D)$", title="Offline design spectrum")
        axes[0].legend()
        axes[1].semilogy(test_vv[:,0], ols_vv_condition, label="OLS")
        axes[1].semilogy(test_vv[:,0], ridge_vv_condition, label="ridge-14")
        axes[1].axvspan(train_vv[:,0].min(), train_vv[:,0].max(), color="0.8")
        axes[1].set(xlabel=r"$V_v$ [MeV]", ylabel=r"$\kappa_2[A(V_v)]$", title="Online conditioning")
        axes[1].legend()
        axes[2].semilogy(test_vv[:,0], relative_l2(ols_vv_wave,test_phi_vv), label="OLS")
        axes[2].semilogy(test_vv[:,0], relative_l2(ridge_vv_wave,test_phi_vv), label="ridge-14")
        axes[2].set(xlabel=r"$V_v$ [MeV]", ylabel="relative L2 error", title="Wavefunction error")
        axes[2].legend()
        plt.show()
        """),
        md(r"""
        ## 3.1 Three-parameter ridge comparison

        We repeat the comparison for the $(V_v,R_v,a_v)$ design. The ridge fit
        reuses the OLS reduced basis and selected predictor coordinates, so any
        change comes only from how the implicit equation coefficients are fit.
        """),
        code(r"""
        ridge_ws3 = ScatteringLROM(
            problem, BasisConfig(size=BASIS_SIZE), MaxVol(count=3, ridge=RIDGE)
        ).train(training_data_ws, precomputed_bases={(0,0.0): model_ws3.basis})
        ridge_test_ws = np.asarray([
            ridge_ws3.predict(row).wavefunction((0,0.0)) for row in test_samples_ws
        ])
        ols_condition_ws = np.asarray([
            emulator_ws3.predict(row).condition_number((0,0.0)) for row in test_samples_ws
        ])
        ridge_condition_ws = np.asarray([
            ridge_ws3.predict(row).condition_number((0,0.0)) for row in test_samples_ws
        ])
        pd.DataFrame({
            "fit":["OLS","ridge-14"],
            "median relative L2":[np.median(relative_l2(lrom_test_ws3,test_phi_ws)),np.median(relative_l2(ridge_test_ws,test_phi_ws))],
            "95% relative L2":[np.quantile(relative_l2(lrom_test_ws3,test_phi_ws),.95),np.quantile(relative_l2(ridge_test_ws,test_phi_ws),.95)],
            "worst condition":[ols_condition_ws.max(),ridge_condition_ws.max()],
            "coefficient norm":[
                np.linalg.norm(np.r_[model_ws3.matrices.ravel(),model_ws3.vectors.ravel()]),
                np.linalg.norm(np.r_[ridge_ws3.channel_models[(0,0.0)].learned.matrices.ravel(),ridge_ws3.channel_models[(0,0.0)].learned.vectors.ravel()]),
            ],
        })
        """),
        code(r"""
        fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
        axes[0].scatter(ols_condition_ws, relative_l2(lrom_test_ws3,test_phi_ws), s=22, alpha=.7, label="OLS")
        axes[0].scatter(ridge_condition_ws, relative_l2(ridge_test_ws,test_phi_ws), s=22, alpha=.7, label="ridge-14")
        axes[0].set(xscale="log",yscale="log",xlabel="online condition number",ylabel="relative L2 error",title="Conditioning versus accuracy")
        axes[0].legend()
        axes[1].boxplot([np.log10(relative_l2(lrom_test_ws3,test_phi_ws)),np.log10(relative_l2(ridge_test_ws,test_phi_ws))],tick_labels=["OLS","ridge-14"])
        axes[1].set(ylabel="log10 relative L2 error",title="Held-out error distribution")
        plt.show()
        """),
        md(r"""
        ## 3.2 Compiled batched online solve

        `LROM_BACKEND=cpu` uses NumPy. `LROM_BACKEND=gpu` uses a JIT-compiled
        JAX solve on a visible GPU; `auto` selects a GPU when available. Only
        the batched reduced systems move to the accelerator—the fitted equation,
        feature construction, S-matrix formula, and observable remain identical.
        """),
        code(r"""
        angles_backend = np.linspace(1,179,90)
        cpu_runtime = get_runtime("cpu")
        ridge_ws3.repack("padded")
        # Warm up NumPy/JAX and then time a whole held-out batch.
        ridge_ws3.cross_sections(test_samples_ws,angles_backend,linear_solver=runtime.solve)
        rows=[]
        for label,engine in [("NumPy CPU",cpu_runtime),(runtime.name,runtime)]:
            elapsed=[]
            for _ in range(8):
                tic=time.perf_counter()
                values=ridge_ws3.cross_sections(test_samples_ws,angles_backend,linear_solver=engine.solve)
                elapsed.append(time.perf_counter()-tic)
            rows.append({"runtime":label,"device":engine.device,"batch cases":len(test_samples_ws),"best milliseconds":1000*min(elapsed),"cases/second":len(test_samples_ws)/min(elapsed)})
        pd.DataFrame(rows)
        """),
    ]
    nb.cells[-1:-1] = extension
    clear(nb)
    nbf.write(nb, path)


def enhance_02():
    path = NOTEBOOKS / "02_elastic_cross_sections.ipynb"
    nb = nbf.read(path, as_version=4)
    if any("RIDGE_CROSS_SECTION_EXTENSION" in c.source for c in nb.cells):
        return
    setup = nb.cells[2].source
    setup = setup.replace(
        "import hashlib",
        "import os, sys\nfrom pathlib import Path\n"
        "ROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
        "sys.path.insert(0,str(ROOT))\nos.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))\n\nimport hashlib",
    )
    setup = setup.replace(
        "BasisConfig, InputSpace, MaxVol, ScatteringLROM, ScatteringProblem,",
        "BasisConfig, InputSpace, MaxVol, ScatteringLROM, ScatteringProblem, get_runtime,",
    )
    setup += """

BACKEND=os.getenv("LROM_BACKEND","cpu")
PRECISION=os.getenv("LROM_PRECISION","double")
runtime=get_runtime(BACKEND,PRECISION)
print(f"online runtime: {runtime.name} on {runtime.device} ({runtime.precision})")
"""
    nb.cells[2].source = setup
    extension = [
        md(r"""
        <!-- RIDGE_CROSS_SECTION_EXTENSION -->
        ## 5. Weak ridge on exactly the same reduced model

        The stability study showed that a tiny ridge can suppress poorly
        identified coefficient combinations. To isolate that choice, the model
        below reuses the default OLS bases, training snapshots, predictor count,
        and max-volume locations. Only the regression solver changes.
        """),
        code(r"""
        RIDGE=1e-14
        ols_default=results[DEFAULT_CONFIGURATION]["emulator"]
        shared_bases={key:value.learned.basis for key,value in ols_default.channel_models.items()}
        ridge_default=ScatteringLROM(
            problem,
            BasisConfig(size=DEFAULT_CONFIGURATION[0]),
            MaxVol(count=DEFAULT_CONFIGURATION[1],minimum_s=MINIMUM_PREDICTOR_S,ridge=RIDGE),
            packing_strategy="padded",
        ).train(training_data,precomputed_bases=shared_bases)
        ridge_train_xs=ridge_default.cross_sections(train_samples,angles,linear_solver=runtime.solve)
        ridge_test_xs=ridge_default.cross_sections(test_samples,angles,linear_solver=runtime.solve)
        ridge_train_error=pointwise_relative_error(ridge_train_xs,fom_train_xs,denominator_floor=ERROR_DENOMINATOR_FLOOR)
        ridge_test_error=pointwise_relative_error(ridge_test_xs,fom_test_xs,denominator_floor=ERROR_DENOMINATOR_FLOOR)
        ols_test_error=results[DEFAULT_CONFIGURATION]["LROM"]["test_median_error"]

        pd.DataFrame({
            "fit":["OLS","ridge-14"],
            "median held-out point error":[np.median(ols_test_error),np.median(ridge_test_error)],
            "95% held-out point error":[np.quantile(ols_test_error,.95),np.quantile(ridge_test_error,.95)],
            "maximum held-out point error":[np.max(ols_test_error),np.max(ridge_test_error)],
        })
        """),
        md(r"""
        ### Design spectrum, coefficient size, and online conditioning

        A rapidly decaying snapshot spectrum is welcome; a rapidly decaying
        **design-matrix** spectrum means several learned-equation coefficient
        combinations have almost the same training residual. We inspect both
        the offline spectrum and the condition numbers of the resulting online
        matrices $A(\omega)$.
        """),
        code(r"""
        diagnostic_channel=(2,2.0)
        learned_ols=ols_default.channel_models[diagnostic_channel].learned
        coords=learned_ols.basis.project(training_data.wavefunctions[diagnostic_channel])
        features=learned_ols.features(training_data.potentials[diagnostic_channel])
        design=np.concatenate([np.einsum("sf,si->sfi",features,coords,optimize=True),-features[...,None]],axis=2).reshape(len(features),features.shape[1]*(coords.shape[1]+1))
        spectrum=np.linalg.svd(design,compute_uv=False)
        condition_rows=[]
        for sample in test_samples:
            for channel in problem.channels:
                condition_rows.append((
                    ols_default.predict(sample).condition_number(channel),
                    ridge_default.predict(sample).condition_number(channel),
                ))
        condition_rows=np.asarray(condition_rows)
        coefficient_norm=lambda model: np.sqrt(sum(np.linalg.norm(v["matrices"])**2+np.linalg.norm(v["vectors"])**2 for v in model.matrices().values()))
        pd.DataFrame({
            "fit":["OLS","ridge-14"],
            "coefficient norm":[coefficient_norm(ols_default),coefficient_norm(ridge_default)],
            "median condition":[np.median(condition_rows[:,0]),np.median(condition_rows[:,1])],
            "99% condition":[np.quantile(condition_rows[:,0],.99),np.quantile(condition_rows[:,1],.99)],
            "maximum condition":[condition_rows[:,0].max(),condition_rows[:,1].max()],
        })
        """),
        code(r"""
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
        axes[0].semilogy(spectrum/spectrum[0],"o-")
        axes[0].axhline(np.sqrt(RIDGE),color="tab:red",ls="--",label=r"$\sqrt{\lambda}$")
        axes[0].set(xlabel="singular-value index",ylabel=r"$\sigma_i(D)/\sigma_1(D)$",title=f"Design spectrum, channel {diagnostic_channel}")
        axes[0].legend()
        axes[1].violinplot([np.log10(condition_rows[:,0]),np.log10(condition_rows[:,1])],showmedians=True)
        axes[1].set_xticks([1,2],["OLS","ridge-14"]); axes[1].set_ylabel("log10 online condition"); axes[1].set_title("All held-out channels")
        axes[2].violinplot([np.log10(np.maximum(ols_test_error,1e-14)),np.log10(np.maximum(np.median(ridge_test_error,axis=1),1e-14))],showmedians=True)
        axes[2].set_xticks([1,2],["OLS","ridge-14"]); axes[2].set_ylabel("log10 median angular error"); axes[2].set_title("Held-out cross sections")
        plt.show()
        """),
        md(r"""
        ### Representative cross sections: OLS versus ridge

        We show the ridge-best, median, and worst held-out cases according to
        their median angular discrepancy. These plots are more informative than
        a single aggregate score when the cross section spans many decades.
        """),
        code(r"""
        ridge_case_error=np.median(ridge_test_error,axis=1)
        order=np.argsort(ridge_case_error)
        chosen=order[[0,len(order)//2,-1]]
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
        for axis,index in zip(axes,chosen):
            axis.semilogy(angles,fom_test_xs[index],color="black",lw=2.2,label="FOM")
            axis.semilogy(angles,results[DEFAULT_CONFIGURATION]["LROM"]["test_xs"][index],label="OLS")
            axis.semilogy(angles,ridge_test_xs[index],"--",label="ridge-14")
            axis.set(xlabel=r"$\theta_{cm}$ [deg]",ylabel=r"$d\sigma/d\Omega$ [mb/sr]",title=f"held-out case {index}")
        axes[0].legend(); plt.show()
        """),
        md(r"""
        ## 6. CPU and compiled GPU deployment

        The public model is unchanged across devices. NumPy solves the stacked
        reduced systems on the CPU. With `LROM_BACKEND=gpu`, JAX compiles the
        batched solve and executes it on a visible GPU. Feature construction and
        the angular observable remain common CPU code, which avoids maintaining
        two scientific implementations.
        """),
        code(r"""
        cpu_runtime=get_runtime("cpu")
        ridge_default.cross_sections(test_samples,angles,linear_solver=runtime.solve)
        timing=[]
        for label,engine in [("NumPy CPU",cpu_runtime),(runtime.name,runtime)]:
            elapsed=[]
            for _ in range(10):
                tic=time.perf_counter(); batch=ridge_default.cross_sections(test_samples,angles,linear_solver=engine.solve); elapsed.append(time.perf_counter()-tic)
            timing.append({"runtime":label,"device":engine.device,"precision":engine.precision,"batch size":len(test_samples),"best ms":1000*min(elapsed),"cross sections/s":len(test_samples)/min(elapsed)})
        pd.DataFrame(timing)
        """),
    ]
    nb.cells[-1:-1] = extension
    clear(nb)
    nbf.write(nb, path)


def build_03():
    cells=[
        md(r"""
        # 03. Curated neutron-elastic data and the global windowed LROM

        This notebook is a transparent audit of the current global workflow.
        It loads the vendored KDUQ fit corpus and post-KD held-out corpus,
        overlays every retained angular distribution with the unmodified
        Koning--Delaroche (KD) prediction, and validates the distributed
        three-window ridge LROM against the independent FOM.

        The black curve is always the current KD FOM. The dashed orange curve
        is the hard-routed LROM evaluated at the same KD parameters. Data points
        retain their reported statistical uncertainties. No calibration is
        performed here: this notebook establishes the baseline to which future
        MAP and posterior calculations must be compared.
        """),
        md(r"""
        ## Model and data provenance

        The data in `data/curated` are an unmodified subset of
        `beykyle/nucleon-nucleus-data` at the commit recorded in
        `PROVENANCE.md`. The pretrained models in `models/three_window` use
        independent bases in the hard energy windows 5–70, 70–135, and
        135–200 MeV. Every training support is wider than its routing interval,
        reducing accidental boundary extrapolation.
        """),
        code(r"""
        import os,sys,time,math
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        ROOT=Path.cwd().resolve()
        if ROOT.name=="notebooks": ROOT=ROOT.parent
        sys.path.insert(0,str(ROOT))
        os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".matplotlib-cache"))

        from lrom import get_runtime,load_three_window_lrom
        from lrom.curated_data import load_neutron_elastic,strict_lrom_selection

        plt.style.use("seaborn-v0_8-whitegrid")
        BACKEND=os.getenv("LROM_BACKEND","cpu")
        PRECISION=os.getenv("LROM_PRECISION","double")
        runtime=get_runtime(BACKEND,PRECISION)
        DATA_ROOT=ROOT/"data"/"curated"
        MODEL_ROOT=ROOT/"models"/"three_window"
        OUTPUT=ROOT/"artifacts"/"real_data_kd_overlays"
        OUTPUT.mkdir(parents=True,exist_ok=True)
        DISPLAY_ANGLES=np.linspace(1,179,179)
        print(f"runtime={runtime.name}, device={runtime.device}, precision={runtime.precision}")
        """),
        md("## 1. Corpus audit and coverage"),
        code(r"""
        raw_points,raw_records=load_neutron_elastic(DATA_ROOT)
        points=strict_lrom_selection(raw_points)
        records=(points[["record_id","corpus","role","target","A","Z","energy_MeV","normalization_fraction","EXFOR_accession"]].drop_duplicates("record_id").reset_index(drop=True))
        pd.DataFrame({
            "records":records.groupby("corpus").size(),
            "points":points.groupby("corpus").size(),
            "isotopes":records.groupby("corpus").target.nunique(),
            "minimum energy":records.groupby("corpus").energy_MeV.min(),
            "maximum energy":records.groupby("corpus").energy_MeV.max(),
        })
        """),
        code(r"""
        fig,axes=plt.subplots(1,3,figsize=(16,4),layout="constrained")
        colors={"kduq":"#2457E6","test":"#D62728"}
        for corpus,group in records.groupby("corpus"):
            axes[0].scatter(group.A,group.energy_MeV,s=24,alpha=.7,label=corpus,color=colors[corpus])
            axes[1].hist(group.energy_MeV,bins=np.linspace(5,200,25),histtype="step",lw=2,label=corpus,color=colors[corpus])
            axes[2].hist(group.A,bins=25,histtype="step",lw=2,label=corpus,color=colors[corpus])
        for edge in (70,135): axes[0].axhline(edge,color="0.4",ls="--")
        axes[0].set(xlabel="mass number A",ylabel=r"$E_{lab}$ [MeV]",title="Experimental records")
        axes[1].set(xlabel=r"$E_{lab}$ [MeV]",ylabel="records",title="Energy coverage")
        axes[2].set(xlabel="mass number A",ylabel="records",title="Isotope coverage")
        for axis in axes: axis.legend()
        plt.show()
        """),
        md("## 2. Load the trusted hard-routed deployment"),
        code(r"""
        deployment=load_three_window_lrom(MODEL_ROOT,DISPLAY_ANGLES,runtime=runtime)
        fom_problem=deployment.models[0].problem
        manifest=pd.read_json(MODEL_ROOT/"training_manifest.json") if False else None
        pd.DataFrame({
            "window":[0,1,2],"core low":deployment.energy_edges[:-1],"core high":deployment.energy_edges[1:],
            "basis size":[14]*3,"potential predictors":[20]*3,"ridge":[1e-14]*3,
            "partial-wave channels":[len(model.problem.channels) for model in deployment.models],
        })
        """),
        md(r"""
        ## 3. Evaluate KD with the FOM and LROM at every experimental record

        For each angular distribution, one union grid contains both the dense
        display angles and every measured angle. This avoids interpolation when
        computing emulator/FOM discrepancies at the data locations.
        """),
        code(r"""
        def normalization_profile(theory,observed,sigma,tau):
            weight=1/np.maximum(sigma,1e-30)**2
            delta=np.sum(theory*(observed-theory)*weight)/(1/tau**2+np.sum(theory**2*weight))
            chi2=np.sum(((observed-(1+delta)*theory)/sigma)**2)+(delta/tau)**2
            return float(delta),float(chi2)

        predictions={}; metric_rows=[]
        fom_seconds=lrom_seconds=0.0
        for record in records.itertuples(index=False):
            group=points[points.record_id==record.record_id]
            measured=group.angle_cm_deg.to_numpy(float)
            angle_grid=np.unique(np.r_[DISPLAY_ANGLES,measured])
            sample={"A":int(record.A),"Z":int(record.Z),"E":float(record.energy_MeV)}
            tic=time.perf_counter(); fom=fom_problem.cross_section(sample,angle_grid); fom_seconds+=time.perf_counter()-tic
            tic=time.perf_counter(); lrom=deployment.models[deployment.route_energy(record.energy_MeV)].cross_section(sample,angle_grid); lrom_seconds+=time.perf_counter()-tic
            indices=np.searchsorted(angle_grid,measured)
            f,l=fom[indices],lrom[indices]
            floor=max(1e-12,1e-6*np.max(f))
            log_error=np.abs(np.log10(np.maximum(l,floor))-np.log10(np.maximum(f,floor)))
            active=f>floor
            relative=np.abs(l[active]-f[active])/f[active] if np.any(active) else np.asarray([np.nan])
            observed=group.cross_section_mb_sr.to_numpy(float)
            reported=group.point_error_mb_sr.to_numpy(float)
            sigma=np.sqrt(reported**2+(.02*observed)**2+(.001*observed.max())**2)
            tau=float(record.normalization_fraction)
            _,chi_fom=normalization_profile(f,observed,sigma,tau)
            _,chi_lrom=normalization_profile(l,observed,sigma,tau)
            window=deployment.route_energy(record.energy_MeV)
            metric_rows.append(dict(record_id=record.record_id,corpus=record.corpus,target=record.target,A=record.A,Z=record.Z,energy_MeV=record.energy_MeV,window=window,points=len(group),median_log10_error=np.median(log_error),p95_log10_error=np.quantile(log_error,.95),median_relative_error=np.nanmedian(relative),chi2_per_point_fom=chi_fom/len(group),chi2_per_point_lrom=chi_lrom/len(group)))
            predictions[record.record_id]=dict(group=group,angles=angle_grid,fom=fom,lrom=lrom)
        metrics=pd.DataFrame(metric_rows)
        print(f"evaluated {len(records)} records / {len(points)} data points")
        print(f"FOM wall time {fom_seconds:.2f} s; LROM wall time {lrom_seconds:.2f} s; scalar speedup {fom_seconds/lrom_seconds:.1f}x")
        """),
        code(r"""
        metrics.groupby(["corpus","window"]).agg(
            records=("record_id","size"),
            median_log10_error=("median_log10_error","median"),
            p95_log10_error=("p95_log10_error",lambda x:np.quantile(x,.95)),
            median_relative_error=("median_relative_error","median"),
            median_chi2_FOM=("chi2_per_point_fom","median"),
            median_chi2_LROM=("chi2_per_point_lrom","median"),
        )
        """),
        code(r"""
        fig,axes=plt.subplots(1,3,figsize=(16,4),layout="constrained")
        for corpus,color in colors.items():
            group=metrics[metrics.corpus==corpus]
            axes[0].scatter(group.energy_MeV,group.median_log10_error,s=24,alpha=.65,label=corpus,color=color)
            axes[1].scatter(group.A,group.median_log10_error,s=24,alpha=.65,label=corpus,color=color)
            axes[2].scatter(group.chi2_per_point_fom,group.chi2_per_point_lrom,s=24,alpha=.65,label=corpus,color=color)
        axes[0].set(xlabel=r"$E_{lab}$ [MeV]",ylabel="median |log10 LROM/FOM|",title="Emulator error versus energy")
        axes[1].set(xlabel="mass number A",ylabel="median |log10 LROM/FOM|",title="Emulator error versus isotope")
        limit=max(axes[2].get_xlim()[1],axes[2].get_ylim()[1]); axes[2].plot([0,limit],[0,limit],color="black",ls="--")
        axes[2].set(xscale="log",yscale="log",xlabel=r"FOM $\chi^2/N$",ylabel=r"LROM $\chi^2/N$",title="Effect on data agreement")
        for axis in axes: axis.legend()
        plt.show()
        """),
        md(r"""
        ## 4. Plot every angular distribution

        The following cell writes paginated PNG files containing every retained
        record. This prevents a 40-page notebook output while still making the
        complete audit easy to inspect. The first page and the worst emulator
        cases are displayed inline.
        """),
        code(r"""
        ordered=records.sort_values(["corpus","A","Z","energy_MeV","record_id"])
        page_size=12
        page_paths=[]
        for page,start in enumerate(range(0,len(ordered),page_size),1):
            subset=ordered.iloc[start:start+page_size]
            fig,axes=plt.subplots(3,4,figsize=(16,11),layout="constrained")
            for axis in axes.flat: axis.set_visible(False)
            for axis,record in zip(axes.flat,subset.itertuples(index=False)):
                axis.set_visible(True); item=predictions[record.record_id]; group=item["group"]
                axis.errorbar(group.angle_cm_deg,group.cross_section_mb_sr,yerr=group.point_error_mb_sr,fmt="o",ms=2.8,color=colors[record.corpus],alpha=.8,label=record.corpus)
                axis.semilogy(item["angles"],item["fom"],color="black",lw=1.7,label="KD FOM")
                axis.semilogy(item["angles"],item["lrom"],color="#E69F00",ls="--",lw=1.2,label="windowed LROM")
                axis.set(title=f"{record.target}, {record.energy_MeV:g} MeV",xlabel=r"$\theta_{cm}$ [deg]",ylabel=r"$d\sigma/d\Omega$ [mb/sr]")
            handles,labels=next(axis.get_legend_handles_labels() for axis in axes.flat if axis.get_visible())
            fig.legend(handles,labels,loc="outside upper center",ncol=3)
            path=OUTPUT/f"all_records_page_{page:02d}.png"; fig.savefig(path,dpi=140); page_paths.append(path)
            if page==1: plt.show()
            else: plt.close(fig)
        print(f"wrote {len(page_paths)} pages to {OUTPUT}")
        """),
        md("## 5. Worst LROM/FOM cases under a microscope"),
        code(r"""
        worst=metrics.nlargest(6,"p95_log10_error")
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained")
        for axis,record in zip(axes.flat,worst.itertuples(index=False)):
            item=predictions[record.record_id]; group=item["group"]
            axis.errorbar(group.angle_cm_deg,group.cross_section_mb_sr,yerr=group.point_error_mb_sr,fmt="o",ms=3,color=colors[record.corpus])
            axis.semilogy(item["angles"],item["fom"],color="black",lw=2,label="KD FOM")
            axis.semilogy(item["angles"],item["lrom"],color="#E69F00",ls="--",label="windowed LROM")
            axis.set(title=f"{record.target}, {record.energy_MeV:g} MeV\np95 log error={record.p95_log10_error:.3g}",xlabel=r"$\theta_{cm}$ [deg]",ylabel=r"$d\sigma/d\Omega$ [mb/sr]")
        axes.flat[0].legend(); plt.show()
        worst[["corpus","target","energy_MeV","window","median_log10_error","p95_log10_error","median_relative_error","chi2_per_point_fom","chi2_per_point_lrom"]]
        """),
        md(r"""
        ## 6. Batched deployment throughput and conditioning

        Calibration asks for many cases at once. The router groups cases by
        energy window and sends the stacked reduced solves to the configured
        runtime. We also scan the maximum channel condition number for a
        stratified subset; this is the online singularity diagnostic, distinct
        from the offline design spectrum.
        """),
        code(r"""
        unique_cases=[{"A":int(r.A),"Z":int(r.Z),"E":float(r.energy_MeV)} for r in records.itertuples(index=False)]
        deployment.partial_wave_s_matrices(unique_cases[:4])
        tic=time.perf_counter(); plus,minus,wave_numbers=deployment.partial_wave_s_matrices(unique_cases); batch_seconds=time.perf_counter()-tic
        print(f"{len(unique_cases)} routed cases in {1000*batch_seconds:.1f} ms = {len(unique_cases)/batch_seconds:.0f} cases/s ({runtime.name})")

        subset_index=np.unique(np.linspace(0,len(records)-1,min(80,len(records)),dtype=int))
        condition_rows=[]
        for index in subset_index:
            record=records.iloc[index]; window=deployment.route_energy(record.energy_MeV); model=deployment.models[window]
            sample={"A":int(record.A),"Z":int(record.Z),"E":float(record.energy_MeV)}
            conditions=[model.predict(sample).condition_number(channel) for channel in model.problem.channels]
            condition_rows.append(dict(corpus=record.corpus,target=record.target,energy_MeV=record.energy_MeV,window=window,maximum_condition=max(conditions)))
        condition=pd.DataFrame(condition_rows)
        condition.groupby("window").maximum_condition.agg(["median",lambda x:np.quantile(x,.95),"max"])
        """),
        code(r"""
        fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
        for window,group in condition.groupby("window"):
            ax.semilogy(group.energy_MeV,group.maximum_condition,"o",alpha=.7,label=f"window {window}")
        for edge in (70,135): ax.axvline(edge,color="0.4",ls="--")
        ax.set(xlabel=r"$E_{lab}$ [MeV]",ylabel="maximum channel condition",title="Online stability on experimental cases")
        ax.legend(); plt.show()
        """),
        md(r"""
        # Conclusions and next checks

        - KD/data disagreement is a property of the baseline optical model;
          LROM/FOM disagreement is emulator error. They must not be conflated.
        - The KDUQ corpus is for fitting; the newer `test` corpus is held out.
        - Hard routing evaluates one local model per energy, while overlapping
          training supports protect window boundaries.
        - Production calibration should repeat these diagnostics over the full
          optical-parameter prior, not only at the central KD curve.
        - If a worst case coincides with a large online condition number, add
          training support or revise that window before expanding the Bayesian
          calculation.
        """),
    ]
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}})
    nbf.write(nb,NOTEBOOKS/"03_curated_data_and_global_lrom.ipynb")


enhance_01(); enhance_02(); build_03()
print("Collaborator notebooks prepared")

