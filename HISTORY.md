Release History
===============

0.3.2 (Nov 2024)
------------------
* BMTI deltaFs error improvements by @charliematteo in https://github.com/sissa-data-science/DADApy/pull/133
* typo fixed by @wildromi in https://github.com/sissa-data-science/DADApy/pull/134
* updated docs and minor fixes to density_advanced by @charliematteo in https://github.com/sissa-data-science/DADApy/pull/136
* Fix DII unittests for newer Python versions by @FelixWodaczek in https://github.com/sissa-data-science/DADApy/pull/137
* Dii optimization cosine decay by @wildromi in https://github.com/sissa-data-science/DADApy/pull/143
* Bmti error dev by @charliematteo in https://github.com/sissa-data-science/DADApy/pull/141
* codecov: update to v4 by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/144
* BID by @acevedo-s in https://github.com/sissa-data-science/DADApy/pull/140
* Add jax implementation of DII by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/146

## New Contributors
* @acevedo-s made their first contribution in https://github.com/sissa-data-science/DADApy/pull/140

**Full Changelog**: https://github.com/sissa-data-science/DADApy/compare/v0.3.1...v0.3.2


0.3.0 (May 2024)
------------------
* bugfix: kstar_gride by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/96
* added cython version for data_overlap by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/97
* Add basic jackknife for information imbalance (WIP) by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/98
* Fix docs by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/99
* Fixed cross NN with periodic box and added tests. by @ollyfutur in https://github.com/sissa-data-science/DADApy/pull/102
* add @ollyfutur to list of contributors by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/103
* Causality dev by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/105
* fix warnings related to float64-32 conversion by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/106
* add n_jobs option to compute_nn_distances  by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/109
* fixes issue 108 by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/110
* Refactor overlap functions and add new test envs by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/111
* fix Gride range_max + return ranks/datasets sizes in ID scaling functions by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/112
* add python 12 compatibility by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/113
* Class imbalance support for return_label_overlap method  by @alexserra98 in https://github.com/sissa-data-science/DADApy/pull/114
* renaming decimation and fraction in ID + cleanupoverlap unreachable code by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/115
* Avoid warning about data type when using float64 by @alexdepremia in https://github.com/sissa-data-science/DADApy/pull/116
* Differentiable Imbalance by @FelixWodaczek in https://github.com/sissa-data-science/DADApy/pull/117
* fix feature_selection unittest by @FelixWodaczek in https://github.com/sissa-data-science/DADApy/pull/118
* Add the BMTI method for density estimation by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/119
* kstar computation is mandatory when calling PAk by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/120
* fix black linting  by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/121
* DII dev by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/122
* Differentiable imbalance ex fix2 by @wildromi in https://github.com/sissa-data-science/DADApy/pull/125
* Added instructions for readthedocs Differentiable Information Imbalance by @wildromi in https://github.com/sissa-data-science/DADApy/pull/126
* information imbalance with automatic subsampling by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/127
* Causality bugfix by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/129
* Fix error generated in tutorial "notebook_on_intrinsicdim_densityest_clustering.ipynb" by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/130
* Pvalues by @imacocco in https://github.com/sissa-data-science/DADApy/pull/128

#### New Contributors
* @ollyfutur made their first contribution in https://github.com/sissa-data-science/DADApy/pull/102
* @alexserra98 made their first contribution in https://github.com/sissa-data-science/DADApy/pull/114
* @FelixWodaczek made their first contribution in https://github.com/sissa-data-science/DADApy/pull/117

**Full Changelog**: https://github.com/sissa-data-science/DADApy/compare/v0.2.0...v0.3.0


0.2.0 (May 2023)
------------------
* PaK: solution for singular likelihood hessian and infinite volume shells by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/77
* Addition clustering adpy v2 and other improvements by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/78
* adding naive outlier detection to mus gride by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/80
* Update reference article from ArXiv to published paper by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/81
* optimizing overlaps with labels by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/82
* fix flake8 errors by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/86
* add initial pyproject.toml by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/88
* update copyright 2021-2022 -> 2021-2023 by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/89
* add kstar-gride id estimator and basic test by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/87
* Develop causality by @vdeltatto in https://github.com/sissa-data-science/DADApy/pull/90
* added Kstar binomial estimator by @imacocco in https://github.com/sissa-data-science/DADApy/pull/93
* 'Return_id_scaling_2NN' now works when the class is initalized with only distance matrices by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/92
* I3D by @imacocco in https://github.com/sissa-data-science/DADApy/pull/83

#### New Contributors
* @vdeltatto made their first contribution in https://github.com/sissa-data-science/DADApy/pull/90

**Full Changelog**: https://github.com/sissa-data-science/DADApy/compare/v0.1.1...v0.2.0


v0.1.1 (July 2022)
------------------
* added a reference to the arxiv package description by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/59
* add warning for datasets with overlapping datapoints by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/60
* add option to save mus in compute_id_2nn by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/62
* docs/usage: duly -> dadapy typo found by @alexdepremia by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/64
* docs: added a new documentation page describing the hands-on tutorial given at SISSA by @AldoGl in https://github.com/sissa-data-science/DADApy/pull/65
* New ID tutorial jupyter notebbok/ small fixed to compute_id_twoNN by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/66
* Updating the examples files by @alexdepremia in https://github.com/sissa-data-science/DADApy/pull/63
* Save mu attribute in GRIDE by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/68
* solution issue #70 in ADP cython and pure_python version by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/74
* added vectorized pak version by @diegodoimo in https://github.com/sissa-data-science/DADApy/pull/72
* Changed dendrogram representation by @alexdepremia in https://github.com/sissa-data-science/DADApy/pull/73
* examples: add jupyter Mobius strip Jupyter notebook by @imacocco and @AldoGl in https://github.com/sissa-data-science/DADApy/pull/75
* examples: add Jupyter notebook with beta-hairpin data analysis by @wildromi and @AldoGl   in https://github.com/sissa-data-science/DADApy/pull/76

0.1.0 (May 2022)
------------------
* First release of DADApy
