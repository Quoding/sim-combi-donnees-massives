This README describes the data file contents accompanying the paper
entitled, "A dataset quantifying polypharmacy in 
the United States" by Quinn & Shah (2017). This README is intended
to be understood with the Data Records section of the paper.

Contact:  Nigam Shah, nigam@stanford.edu

====================================================================
OVERVIEW
====================================================================
There are 3 data folders:
  Data Record 1: Drug ingredient combinations
  Data Record 2: Drug class combinations
  Data Record 3: Drug mappings

====================================================================
FILE MANIFEST
====================================================================
Data Record 1:  (drug ingredient combinations)
  FORMAT: tab-delimited; 
  COLUMN CONTENTS, for N-drugs:
  col 1 to N: drug_name
  col N+1 to N+5: atleast_exposure_count, exact_exposure_count, estimate_drug_combo_cost_per_day, fraction_all_windows, fraction_exact, (for N>1) observe_per_expect_1s, (for N>2) observe_per_expect_N1

  db_drugs_1s.tsv
  db_drugs_2s.tsv
  db_drugs_3s.tsv
  db_drugs_4s.tsv
  db_drugs_5s.tsv


Data Record 2: (drug class combinations)
  FORMAT: tab-delimited; 
  COLUMN CONTENTS, for N-atc-drug-classes:
  col 1 to N: atc_code
  col N+1 to 2*N: atc_name
  col 2*N+1 to 2*N+4: atleast_exposure_count, exact_exposure_count, fraction_all_windows, fraction_exact, (for N>1) observe_per_expect_1s, (for N>2) observe_per_expect_N1

  db_atc_classes_1s.tsv
  db_atc_classes_2s.tsv
  db_atc_classes_3s.tsv
  db_atc_classes_4s.tsv
  db_atc_classes_5s.tsv

  
Data Record 3: (drug mappings)
  FORMAT: tab-delimited;  
  COLUMN CONTENTS: drug_name, RxCUI, UMLS_CUI, DrugBankID, atc_code, atc_name, estimate_drug_cost_per_day

  drug_mappings_ingredients.tsv

  column contents: atc_class, atc_class_name

  drug_mappings_classes.tsv

