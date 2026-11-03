# Complete published baseline rows plus CES-FL L_A/L_H

All displayed values are100 times native fractions. F_FPR is summed disparity, not a probability. Published baseline rows are reference results transcribed from Ding et al., arXiv2511.10150v5, Tables1/2. Our Xception rows are seed17, one original validation-selected checkpoint per arm, ten downstream epochs; L_A selected epoch3, L_H epoch1. Stage budgets, preprocessing and checkpoint/support policies differ from the paper. No new Ding computation, inference, calibration or model selection was performed to create these tables. Same layout does not establish protocol equivalence. No bold best-cell ranking or pooled win count is used. Paper: https://arxiv.org/html/2511.10150v5. [Uncertainty and support](../UNCERTAINTY.md).

[Table1 fixed0.5](TABLE_1_FIXED05.md) · [Table2 fixed0.5](TABLE_2_FIXED05.md) · [Table1 source-VAL10%](TABLE_1_SOURCE_VAL10.md) · [Table2 source-VAL10%](TABLE_2_SOURCE_VAL10.md) · [LaTeX](TABLES_1_2.tex)

Every table includes all seven published baseline/method rows in each block followed by L_A and L_H. Historical local Ding experiments and unimplemented rows are deliberately left in the original separate report; these files compare published results with the current two models.

At fixed0.5 on FF++: L_A FPR9.365%, FNR3.878%; L_H FPR4.319%, FNR7.070%. At source-validation10%: L_A TEST FPR11.305%, L_H10.688%. A fairness change across unequal operating rates must be considered with FNR and utility.

Primary joint-group uncertainty uses990/1000 usable replicates for FF++,608 for DFDC,587 for Celeb-DF,957 for DFD; retain900minimum. All report intervals are conditional on the single trained seed. DFDC/Celeb-DF joint estimates fail that support requirement. The paper training protocol and ours are not fully matched.

No post-TEST tuning or selective threshold choice is authorized by this table request.
