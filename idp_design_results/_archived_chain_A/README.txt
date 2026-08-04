Archived results generated before the chain A/B fix (2026-05-31).

Issue: Several scripts defaulted to scaffold_chain='A' but 5IMK's nanobody is on chain B.
- ad_design_demo.py used CDR_SPEC='A:26-33,...' — designed on VSIG4 antigen instead of nanobody
- run_idp_design.py defaulted --scaffold_chain A
- run_alzheimers_design.py hardcoded region_spec "A:26-33,..."

Files archived here were generated before these fixes.
For correct results, re-run with: python run_design.py --scaffold_chain B
