"""Runtime diagnostics for protein suitability in BFN confidence/design pipeline.

Detects out-of-distribution inputs that would produce unreliable confidence estimates:
  - Length < 50 or > 250 (training distribution range)
  - Low pLDDT / ipTM (potential IDP or disordered regions)
  - High PAE (predicted structural error)
"""

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Flag:
    message: str
    severity: Severity
    detail: str = ""


@dataclass
class DiagnosticReport:
    flags: list = field(default_factory=list)
    length: int = 0
    mean_plddt: float = 0.0
    iptm: float = 0.0
    mean_pae: float = 0.0
    is_in_distribution: bool = True
    idp_likelihood: str = "unknown"  # "low", "possible", "high"

    def has_warnings(self):
        return any(f.severity in (Severity.WARNING, Severity.ERROR) for f in self.flags)

    def has_errors(self):
        return any(f.severity == Severity.ERROR for f in self.flags)


LENGTH_MIN = 50
LENGTH_MAX = 250
PLDDT_IDP_THRESHOLD = 0.55
PLDDT_LOW_THRESHOLD = 0.60
IPTM_IDP_THRESHOLD = 0.20
IPTM_LOW_THRESHOLD = 0.35
PAE_HIGH_THRESHOLD = 12.0
PAE_WARN_THRESHOLD = 8.0


def diagnose_protein(length, mean_plddt=None, iptm=None, mean_pae=None,
                     idp_annotated=False):
    """Run all checks and return a DiagnosticReport.

    Args:
        length: number of residues
        mean_plddt: mean per-residue pLDDT (0-1)
        iptm: predicted ipTM (0-1)
        mean_pae: mean pairwise PAE (Angstroms)
        idp_annotated: if True, protein is known IDP from DisProt or similar
    """
    report = DiagnosticReport(length=length)
    if mean_plddt is not None:
        report.mean_plddt = mean_plddt
    if iptm is not None:
        report.iptm = iptm
    if mean_pae is not None:
        report.mean_pae = mean_pae

    # --- Length checks ---
    if length < LENGTH_MIN:
        report.flags.append(Flag(
            message=f"Protein length ({length} aa) is below the training minimum ({LENGTH_MIN} aa)",
            severity=Severity.WARNING,
            detail=("Positional embeddings are out-of-distribution. "
                    "pLDDT and ipTM estimates will be unreliable. "
                    "Consider using a longer construct or a different tool.")
        ))
        report.is_in_distribution = False

    if length > LENGTH_MAX:
        report.flags.append(Flag(
            message=f"Protein length ({length} aa) exceeds the training maximum ({LENGTH_MAX} aa)",
            severity=Severity.WARNING,
            detail=("The model was trained on L=50-250. "
                    "Confidence estimates for longer proteins are extrapolated and unreliable. "
                    "Consider splitting into individual domains for evaluation.")
        ))
        report.is_in_distribution = False

    # --- pLDDT checks ---
    if mean_plddt is not None:
        if mean_plddt < PLDDT_IDP_THRESHOLD:
            report.flags.append(Flag(
                message=f"Very low mean pLDDT ({mean_plddt:.3f}) — "
                        f"this structure may be intrinsically disordered (IDP)",
                severity=Severity.WARNING,
                detail=("BFN confidence heads were trained on folded globular proteins. "
                        "pLDDT/ipTM estimates for IDP-like proteins are unreliable. "
                        "Design results should be validated experimentally.")
            ))
            report.idp_likelihood = "high"
        elif mean_plddt < PLDDT_LOW_THRESHOLD:
            report.flags.append(Flag(
                message=f"Low mean pLDDT ({mean_plddt:.3f}) — borderline structural quality",
                severity=Severity.INFO,
                detail="Confidence estimates may have reduced accuracy in this range."
            ))
            if report.idp_likelihood == "unknown":
                report.idp_likelihood = "possible"

    # --- ipTM checks ---
    if iptm is not None:
        if iptm < IPTM_IDP_THRESHOLD:
            report.flags.append(Flag(
                message=f"Very low ipTM ({iptm:.3f}) — "
                        f"global fold confidence is poor (potential IDP/disorder)",
                severity=Severity.WARNING,
                detail="ipTM < 0.20 typically indicates disordered or unstable folds."
            ))
            if report.idp_likelihood == "unknown":
                report.idp_likelihood = "possible"
        elif iptm < IPTM_LOW_THRESHOLD:
            report.flags.append(Flag(
                message=f"Low ipTM ({iptm:.3f}) — below-average fold confidence",
                severity=Severity.INFO,
                detail="The model's training data had ipTM range 0.16-0.49. "
                       "Values in this range are within training distribution but on the low end."
            ))

    # --- PAE checks ---
    if mean_pae is not None:
        if mean_pae > PAE_HIGH_THRESHOLD:
            report.flags.append(Flag(
                message=f"High mean PAE ({mean_pae:.1f} A) — "
                        f"large predicted structural uncertainty",
                severity=Severity.WARNING,
                detail="PAE > 12 A suggests significant structural uncertainty. "
                       "Design results for these regions should be treated with caution."
            ))
        elif mean_pae > PAE_WARN_THRESHOLD:
            report.flags.append(Flag(
                message=f"Moderate mean PAE ({mean_pae:.1f} A) — above-average uncertainty",
                severity=Severity.INFO,
                detail="PAE > 8 A indicates some structural uncertainty."
            ))

    # --- Known IDP ---
    if idp_annotated:
        report.flags.append(Flag(
            message="This protein is annotated as IDP in DisProt — "
                    "confidence estimates are known to be unreliable on disordered proteins",
            severity=Severity.WARNING,
            detail=("IDP blindness: BFN pLDDT shows Pearson r=0.21 on disordered regions "
                    "vs r=0.48 on folded regions. Design results should be validated "
                    "experimentally and treated as exploratory.")
        ))
        report.idp_likelihood = "high"

    # --- Final classification ---
    if report.idp_likelihood == "unknown" and not report.flags:
        report.idp_likelihood = "low"

    return report


def format_report(report):
    """Format a DiagnosticReport as a user-facing string."""
    lines = []
    emoji = {Severity.ERROR: "ERR", Severity.WARNING: "WARN", Severity.INFO: "INFO"}

    for f in report.flags:
        lines.append(f"[{emoji[f.severity]}] {f.message}")

    if report.is_in_distribution and not lines:
        lines.append("[OK] Protein is within training distribution.")

    if report.idp_likelihood in ("possible", "high"):
        lines.append(
            f"[IDP] Disorder likelihood: {report.idp_likelihood}. "
            "Confidence estimates may be unreliable."
        )

    return "\n".join(lines)
