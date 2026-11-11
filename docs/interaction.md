# Interaction analysis



`hp.stats.interaction()` tests interactions between two predictor blocks, called
A and B. Block A comes from an `HLAdata` object. Block B can come from the same
object, an external genotype DataFrame, or selected covariate columns.
`hp.interaction()` is the same function. Results are returned as a pandas
DataFrame and are not saved automatically.

For HLA amino acid position interactions, use `mode="one_vs_block_omnibus"` to
test an AA position jointly against each partner predictor. Use
`mode="pairwise"` to test individual encoded genotype-column pairs.

## G×G example: AA positions and external genotypes

Partner genotype data can be read into a pandas DataFrame with `hp.read_genotype()`
and passed to the interaction analysis via `partner_kind="DF"` option.
The reader supports PLINK PGEN, BED/BIM/FAM and RAW files; see
[Reading PLINK genotype data](genotype_dataframe.md) for formats, allele counting,
variant selection and RAW column naming.

With an existing `hladat`, an IID-indexed phenotype Series `phenotype`, and an
optional IID-indexed covariate DataFrame `covar`:

```python
import hapy as hp

partners = hp.read_genotype(
    "test_data/chr7", format="pgen", variants=["rs1", "rs2"],
)
config = hp.stats.InteractionConfig(
    model_type="logit",
    mode="one_vs_block_omnibus",
    n_jobs=1,
)
results = hp.stats.interaction(
    hladat,
    config,
    kind="AA",
    partner_kind="DF",
    partner_data=partners,
    y=phenotype,
    covar=covar,
)
results.to_csv("aa_genotype_interactions.txt", sep="\t", index=False)
```

Replace the file path and variant IDs with your own. The `AA` block must already
be loaded in `hladat`; see [Association analysis](association.md) for loading AA
data and preparing phenotypes and covariates. For RAW or bfile partners, change
`format` in `read_genotype()` accordingly.

Each column in `partners` is tested as an partner against each eligible AA
position. This compares models with and without the partner-by-AA interaction
terms, while retaining both main effects. It does not combine separate
association P-values.

## Interaction naming

Use `kind` for the focal HLA genotype block and `partner_kind` for the interacting
block. 


### Choosing the blocks

| Parameter | Options | Meaning |
|---|---|---|
| `kind` | `"AA"`, `"HLA"`, `"SNP"` | Loaded genotype block in `hladat`. |
| `partner_kind` | `"AA"`, `"HLA"`, `"SNP"` | Another loaded genotype block in `hladat`. |
| `partner_kind="DF"` | `partner_data` required | External IID-indexed predictor DataFrame, this includes output from `read_genotype()`. |
| `partner_kind="COV"` | `covar` and `cov_block_cols` required | Selected covariate columns from the covariate DataFrame are the interacting predictors. |

`hladat` stores the HLA genotype data. A partner DataFrame is separate from
that object: it must be supplied as `partner_data`, not as `hladat`.

## Interaction modes

### `mode="pairwise"`

Use this mode for HLA allele or SNP pairs, or individual encoded AA residue pairs.

For each selected A column and B column, fit:

```text
Full model: Y ~ baseline covariates + A + B + A:B
```

The reported `P-value` tests the interaction coefficient in the full model. For
logistic regression this is a two-sided Wald z-test; for linear regression it
is a two-sided t-test. `Beta` is the signed interaction coefficient.

```python
config = hp.stats.InteractionConfig(model_type="logit", mode="pairwise")
pairwise = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="DF",
    partner_data=partners, y=phenotype, covar=covar,
)
```

### `mode="one_vs_block_omnibus"`

`"AA"` must be selected on at least one side: `kind="AA"` or `partner_kind="AA"`.
If both sides are AA, the current runner treats the partner-side position as
the omnibus block and tests it against each focal encoded column.

```text
Null model: Y ~ baseline covariates + partner + AA block
Full model: Y ~ baseline covariates + partner + AA block
               + partner:AA_column_1 + partner:AA_column_2 + ...
```

The likelihood-ratio test compares the full and null models on the same
complete-case samples. `P-value` tests the interaction terms jointly; it does not
test the AA or partner main effect. A one-column AA block is also eligible.


Keep predictor names distinct from baseline covariate names. Pairwise mode
handles identical names on the two interacting sides internally. For omnibus
AA–AA tests, overlapping encoded names can still collide: selecting different
position IDs alone does not guarantee distinct model-column names.

## `interaction()` parameters

```python
results = hp.stats.interaction(
    hladat,
    config,
    kind="AA",
    partner_kind="DF",
    famfile=None,
    covar=None,
    y=None,
    variant_filter=None,
    pair_filter=None,
    cov_block_cols=None,
    baseline_covar_cols=None,
    partner_data=partners,
    condition_on=None,
    residue=False,
    verbose=True,
    use_progress_bar=False,
    print_every=500,
)
```

| Parameter | Type | Description |
|---|---|---|
| `hladat` | `HLAdata` | Genotype object containing HLA genotype information |
| `config` | `hp.stats.InteractionConfig` | Model, interaction mode and parallel settings. |
| `kind`, `partner_kind` | `str` | Required block types; see the block table above. |
| `famfile` | `pandas.DataFrame` or `None` | Optional FAM table with `IID`, and `PHENO` when `y` is omitted. Required for an unindexed phenotype array. |
| `covar` | `pandas.DataFrame`, `str`, `Path` or `None` | IID-indexed covariates, or a CSV path with sample IDs in the first column. Load whitespace-delimited files explicitly with pandas. Required for `partner_kind="COV"`. |
| `y` | `pandas.Series`, one-dimensional array-like or `None` | IID-indexed phenotype Series, array aligned to FAM row order, or `None` to use `famfile["PHENO"]`. |
| `variant_filter` | callable or `None` | Select HLA-side variants by ID or reader metadata before genotype blocks are constructed. Reuse the standard variant/gene filter helpers. Does not filter partner-side predictors or conditioning adjustment terms. You can filter partner side with the pandas indexing. |
| `pair_filter` | callable or `None` | Function receiving a candidate interaction context and returning `True` to keep or `False` to exclude it. See examples below. |
| `cov_block_cols` | list of strings or `None` | Covariate columns to test as predictors. Required for `partner_kind="COV"`; do not pass for `partner_kind="DF"`. |
| `baseline_covar_cols` | list of strings or `None` | Covariate columns always included as adjustment terms. By default, all covariates are used, except the tested covariates when `partner_kind="COV"`. `[]` selects no baseline covariates. |
| `partner_data` | `pandas.DataFrame` or `None` | Required for `partner_kind="DF"`. Rows are sample IIDs and columns are partner predictors. Use numeric dosages or explicitly numeric indicators. |
| `condition_on` | `str`, sequence of strings or `None` | Same-kind A-side variant IDs added as adjustment terms. Tasks testing a listed A variant are excluded. |
| `residue` | `bool`, default `False` | In pairwise mode, include the QC-passing reference AA residue when available. Requires AA on at least one side; other modes raise `ValueError`. Existing non-reference tests are retained. |
| `verbose` | `bool`, default `True` | Print scheduled task counts, execution settings, progress and timing. |
| `use_progress_bar` | `bool`, default `False` | Use tqdm when available and `verbose=True`; otherwise use periodic progress messages. |
| `print_every` | `int`, default `500` | Completed tasks between periodic progress messages. |

### `InteractionConfig`

| Parameter | Type | Description |
|---|---|---|
| `model_type` | `"logit"` or `"linear"` | Required. Logistic regression for binary outcomes or ordinary least squares for continuous outcomes. |
| `mode` | `"pairwise"` or `"one_vs_block_omnibus"` | Default `"pairwise"`. Choose explicitly for AA position-level interaction tests. |
| `n_jobs` | `int`, default `1` | Number of workers; 1 runs sequentially. |
| `backend` | `"process"` or `"thread"`, default `"process"` | Parallel execution method. Threads can be used in Jupyter notebooks. |
| `batch_size` | `int`, default `32` | Interaction tasks submitted in each batch. |
| `chunksize` | `int`, default `8` | Maximum number of batch futures in flight. |

Process execution requires an `if __name__ == "__main__":` script guard.

## Samples and covariates

The simplest input is an IID-indexed phenotype Series and an optional
IID-indexed covariate DataFrame. Samples are matched by ID, not table row order.
IDs are converted to strings and must be unique and non-missing. Read IDs as
strings before pandas can discard leading zeros.

Without `famfile`, the phenotype Series defines the target sample IDs. With
`famfile`, its `IID` column defines the target IDs and a supplied Series is
aligned to them. An array-like phenotype instead follows FAM row order.

If the observed non-missing phenotype values are exactly `{1, 2}`, `hapy` recodes
1 to control (0) and 2 to case (1). This recoding happens during phenotype
preparation, before model selection. For logistic regression, otherwise supply
0/1 values and represent missing outcomes as `NaN` rather than numeric sentinel
values such as -9. Use `model_type="linear"` for continuous phenotypes.

Numeric baseline covariates enter as continuous predictors; string/object and
categorical baseline covariates use categorical model terms. Column names
containing spaces or punctuation are quoted for formula construction. An
intercept is included automatically.

For each interaction, samples missing phenotype, a fitted genotype column,
a model covariate or a conditioning genotype are excluded. Sample counts can
differ between interaction tests. `hapy` does not impute these missing values.

## Interacting with a covariate: AA positions × SEX

Use `partner_kind="COV"` to test AA positions against columns already in your covariate
table. `cov_block_cols` selects the interacting columns; all other covariates
are baseline adjustment terms by default. There is no need to create a separate
partner DataFrame.

With SEX in the IID-indexed `covar` table and an IID-indexed phenotype Series
`pheno[2]`:

```python
config = hp.stats.InteractionConfig(
    model_type="logit",
    mode="one_vs_block_omnibus",
    n_jobs=12,
    backend="thread",
)
results = hp.stats.interaction(
    hladat,
    config,
    kind="AA",
    partner_kind="COV",
    covar=covar,
    y=pheno[2],
    cov_block_cols=["SEX"], # must be passed as a list
    verbose=True,
    use_progress_bar=False,
    print_every=500,
)
```

For each AA position, the models are:

```text
Null: Y ~ other covariates + SEX + AA block
Full: Y ~ other covariates + SEX + AA block + SEX:AA_column_1 + ...
```

### Coding the interacting covariate

For a binary SEX predictor, numeric 0/1 coding is convenient. Binary categorical
SEX is also supported in the omnibus mode shown above. No assumption about
which code represents which sex is made by the interaction function.

For `mode="pairwise"`, use numeric interacting predictors, such as a 0/1 binary
indicator. Categorical predictors expand into formula coefficients that the
current pairwise extraction does not handle generally. Baseline covariates can
still be categorical in either mode.

### Several interacting covariates

```python
results = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="COV",
    covar=covar, y=ra[2],
    cov_block_cols=["SEX", "exposure"],
)
```


Use `baseline_covar_cols` to name the desired baseline terms explicitly, for
example `["age", "PC1", "PC2"]`. Do not add the tested partner to that list: its
main effect is already included. To avoid unrelated missing-value exclusions,
pass a covariate table containing the selected interacting predictors and the
adjustment columns you need.

Conditioning with `condition_on` works with covariate interactions too; those
genotype columns enter as additional baseline adjustment terms. Covariate B
blocks are identified as `COV_BLOCK` in filter contexts; select a particular
exposure using `partner_col` in omnibus mode or `b_col` in pairwise mode.

## Selecting interaction tests

### Filter HLA variants before constructing genotype blocks

Use `variant_filter` to select HLA-side positions, alleles or SNPs before their
genotype matrices are built. This avoids preparing excluded variants.

```python
variant_filter = hp.stats.filters.only_genes_variant(["DRB1", "DQB1"])
filtered = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="COV",
    covar=covar, cov_block_cols=["SEX"], y=phenotype,
    variant_filter=variant_filter,
)
```

Or select exact position IDs:

```python
variant_filter = hp.stats.filters.only_variants_variant([
    "AA_DRB1_11_32552129",
    "AA_DRB1_13_32552123",
])
```

Pass the helper result as `variant_filter`, not a bare list. Use IDs present
in your loaded data. Gene selection uses reader metadata, so no genotype block
is needed to identify the gene.

### Select individual interactions after constructing blocks

`pair_filter` selects tests after the genotype blocks have been built. For
external genotypes, use `read_genotype(..., variants=[...])` first to reduce
reading and memory use; `pair_filter` does not prevent the reader loading data.

For example, select AA genes on A:

```python
pair_filter = hp.stats.filters.only_genes_pair(genes_a=["DRB1", "DQB1"])
filtered = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="DF",
    partner_data=partners, y=phenotype, covar=covar,
    pair_filter=pair_filter,
)
```

A custom omnibus filter can select an AA position and external partner:

```python
def selected_interaction(ctx):
    return (
        ctx["a_id"] == "AA_DRB1_11_32552129"
        and ctx["partner_col"] == "rs1"
    )

filtered = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="DF",
    partner_data=partners, y=phenotype, covar=covar,
    pair_filter=selected_interaction,
)
```

## Conditional interaction analysis

`condition_on` adjusts for variants from the HLA block selected by `kind`,
matching the argument name used by standard association analysis.

```python
conditional = hp.stats.interaction(
    hladat, config, kind="AA", partner_kind="DF",
    partner_data=partners, y=phenotype, covar=covar,
    condition_on="AA_DRB1_231_32548054",
)
```

Conditioning variant IDs must exist in the A block. Their eligible genotype
columns enter as baseline adjustment terms, and tasks
where the tested A variant is a conditioning variant are excluded. They are not
additional interacting predictors. This option adjusts for A-side variants
only; it does not automatically remove tasks where the same ID occurs on B.
The current interaction runner omits conditioning blocks with no eligible
columns, so a requested empty block does not contribute adjustment terms.



## Reading the interaction results

Each row represents a tested predictor pair in `pairwise` mode, or an AA
position–partner test in `one_vs_block_omnibus` mode. Output columns depend on
the data types and mode. Entirely missing columns may be omitted. Interaction
results do not include a `TEST` column; the test is determined by `config.mode`.

### Variant information and frequencies

| Column | Meaning |
|---|---|
| `VARIANT`, `GENE`, `AA_POS` | Tested variant identifier, gene and amino acid sequence position, where available. In omnibus mode, these describe the AA position being tested jointly, including when it is on the partner side. |
| `Amino_Acids`, `Ref_AA`, `AAcount` | Available amino acid labels, reference residue and allele count, where reported. Additional residue results use `Ref_AA="tested: X"` to identify the tested residue. |
| `Variant_col` | Selected encoded focal genotype column in pairwise analysis when a variant contains multiple tested columns. |
| `Partner_col` | Interacting genotype or covariate column. In omnibus mode, this is the single predictor tested against the AA block. |
| `Partner_*` annotations | Available information about the other block, such as gene, amino acid position and reference residue. An additional partner-side reference test uses `Partner_Ref_AA="tested: X"`. |
| `AA_AF` | Amino acid frequencies on the final modelling samples, formatted as `F=0.2;S=0.8`. Additional reference-residue rows report the tested residue's frequency. |
| `HLA_AF`, `MAF` | Focal HLA allele frequency or SNP minor allele frequency on the final modelling samples, where applicable. |
| `Partner_AF` | Frequency of the selected genotype partner, calculated as mean dosage divided by two. External `partner_data` uses this genotype-dosage convention. |
| `Partner_mean` | Mean of a numeric interacting covariate. For a numeric 0/1 predictor, this is the proportion coded 1. |
| `Partner_proportion` | Proportion in the second category for a binary categorical or boolean interacting covariate. Check the category ordering when interpreting this value. |

Frequencies and the selected partner's summary use the samples included in that
interaction model and can differ between tests because of missing data. AA
dosage frequencies are calculated without renormalising the remaining residues
after QC filtering. Copied partner annotation fields are distinct from the
selected predictor's `Partner_AF`/mean/proportion summary.

### Pairwise interaction statistics

For the model:

```text
Y ~ baseline covariates + variant + partner + variant:partner
```

the primary result tests the `variant:partner` coefficient.

| Column | Meaning |
|---|---|
| `P-value` | Two-sided interaction coefficient P-value. |
| `Statistic` | t-statistic for linear regression or Wald z-statistic for logistic regression. |
| `Beta`, `StdErr` | Interaction coefficient and its standard error. |
| `CI-lower`, `CI-upper` | Lower and upper limits of the interaction coefficient's 95% confidence interval. |
| `Variant_p`, `Variant_Beta` | Focal variant main-effect P-value and coefficient from the full interaction model. |
| `Partner_p`, `Partner_Beta` | Partner main-effect P-value and coefficient from the full interaction model. |

For linear regression, `Beta` describes how the effect of one predictor changes
with a one-unit increase in the other predictor. For logistic regression,
coefficients and confidence intervals are on the log-odds scale. Applying
`np.exp()` to the interaction coefficient gives a ratio of odds ratios, rather
than an overall disease odds ratio.

Main effects are conditional on the other interacting predictor being zero.
For example, `Variant_Beta` describes the variant effect when the partner equals
zero. These are not estimates from separate association models without interaction.

### Omnibus interaction statistics

`one_vs_block_omnibus` compares models with and without all partner-by-AA
interaction terms, retaining the same main effects and adjustment covariates.

| Column | Meaning |
|---|---|
| `P-value` | Likelihood-ratio P-value testing the interaction terms jointly. |
| `Statistic` | Likelihood-ratio statistic: twice the difference between the full and null log-likelihoods. |
| `DoF` | Difference in fitted model degrees of freedom between the full and null models. |

This tests whether the AA position interacts with the partner. It does not
separately test either main effect. The omnibus result does not report a single
interaction coefficient or confidence interval because several interaction
coefficients may be tested jointly. It also does not identify which residue
drives the result.

### Sample counts and fitting information

| Column | Meaning |
|---|---|
| `N_used` | Final number of complete-case samples used in the model. |
| `N_case`, `N_control` | Numbers of cases and controls among the modelling samples for binary outcomes. |
| `N_missing_Y` | Number of rows excluded because the phenotype was missing. May be omitted when zero throughout the table. |
| `N_dropped_other` | Further exclusions because genotype, covariate or conditioning information was missing. May be omitted when zero throughout the table. |
| `INFO` | Missing for fits passing the explicit checks; otherwise describes a fitting problem or absence of complete-case samples. The column can be omitted if entirely missing. |

Fit messages can include `not converged`, `non-finite coefficients`,
`non-finite standard errors` or `non-finite log-likelihood`. Omnibus messages
identify whether the null or full model failed. Failed checks leave statistical
estimates missing; empty statistic columns can remain in tables containing failures.

Model warnings are not automatically copied into `INFO`. A missing `INFO`
value means the implemented checks passed; it does not establish that every
model assumption is satisfied. Unexpected fitting exceptions can still stop
analysis. Genotype blocks with no eligible columns may produce no scheduled
interaction rows, rather than a row with a fitting message.

## Saving and plotting

`N_used` reports the final number of complete-case samples for each test.

```python
results.to_csv("aa_genotype_interactions.txt", sep="\t", index=False)
ranked = results.dropna(subset=["P-value"]).sort_values("P-value")
```

For AA position-level interaction
plots, see [Visualisation](visualisation.md). Supply the omnibus AA ID column
(`VARIANT`) and the appropriate partner identifiers or genomic coordinates.
External partner IDs such as `rs1` do not themselves provide genomic positions;
supply a position column when an ID cannot be parsed.


## Optional reference-residue interactions

With AA on either side, set `residue=True` and `config.mode="pairwise"` to include the QC-passing reference residue without duplicating existing tests.

```python
results = hp.stats.interaction(
    hladat, hp.stats.InteractionConfig(model_type="logit", mode="pairwise", n_jobs=4),
    kind="AA", partner_kind="DF", partner_data=genotypes,
    y=phenotype, covar=covar, residue=True,
)
```

The existing model remains `Y ~ covariates + residue + partner + residue:partner`. The option requires `config.mode="pairwise"` and AA on at least one side; other modes raise `ValueError`. `condition_on`, variant/pair filtering and parallel execution remain available. Position-pair tasks keep related residue contrasts together within the existing workers. When all relevant genotype columns share a missingness pattern, their tests reuse one ABT; differing patterns retain per-combination complete-case filtering.

For complementary biallelic dosages `D_B = 2 - D_A`, an eligible existing fit can be reused. Reversing the variant residue negates its main-effect coefficient and the interaction coefficient, and changes the partner main-effect coefficient to `old_partner_beta + 2 * old_interaction_beta`. The partner main-effect P-value is recalculated using the fitted covariance, rather than copied. Reversing the AA partner applies the corresponding transformation to the other main effect. Reversing both AA sides applies both transformations. Confidence intervals and signed interaction statistics are transformed consistently.
