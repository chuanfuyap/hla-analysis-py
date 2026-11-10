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
and passed to the interaction analysis via `b_kind=DF` option.
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
    a_kind="AA",
    b_kind="DF",
    block_b_df=partners,
    y=phenotype,
    covar=covar,
)
results.to_csv("aa_genotype_interactions.txt", sep="\t", index=False)
```

Replace the file path and variant IDs with your own. The `AA` block must already
be loaded in `hladat`; see [Association analysis](association.md) for loading AA
data and preparing phenotypes and covariates. For RAW or bfile partners, change
`format` in `read_genotype()` accordingly.

Each column in `partners` is tested as an anchor against each eligible AA
position. This compares models with and without the anchor-by-AA interaction
terms, while retaining both main effects. It does not combine separate
association P-values.

## Choosing the blocks

| Parameter | Options | Meaning |
|---|---|---|
| `a_kind` | `"AA"`, `"HLA"`, `"SNP"` | Loaded genotype block in `hladat`. |
| `b_kind` | `"AA"`, `"HLA"`, `"SNP"` | Another loaded genotype block in `hladat`. |
| `b_kind="DF"` | `block_b_df` required | External IID-indexed predictor DataFrame, including output from `read_genotype()`. |
| `b_kind="COV"` | `covar` and `cov_block_cols` required | Selected covariate columns are the interacting B predictors. |

`hladat` stores the HLA/AA genotype data. A partner DataFrame is separate from
that object: it must be supplied as `block_b_df`, not as `hladat`.

## Interaction modes

### `mode="pairwise"`

This is used for HLA (allele) and SNP data type.

For each selected A column and B column, fit:

```text
Full model: Y ~ baseline covariates + A + B + A:B
```

The reported `I_p` tests the interaction coefficient in the full model. For
logistic regression this is a two-sided Wald z-test; for linear regression it
is a two-sided t-test. `I_coef` is the signed interaction coefficient.

```python
config = hp.stats.InteractionConfig(model_type="logit", mode="pairwise")
pairwise = hp.stats.interaction(
    hladat, config, a_kind="AA", b_kind="DF",
    block_b_df=partners, y=phenotype, covar=covar,
)
```

### `mode="one_vs_block_omnibus"`

At least one side must be `"AA"`. One side contributes an AA-position genotype
block and the other contributes one anchor column:

```text
Null model: Y ~ baseline covariates + anchor + AA block
Full model: Y ~ baseline covariates + anchor + AA block
               + anchor:AA_column_1 + anchor:AA_column_2 + ...
```

The likelihood-ratio test compares the full and null models on the same
complete-case samples. `LR_p` tests the interaction terms jointly; it does not
test the AA or anchor main effect. A one-column AA block is also eligible.

If only A is AA, A is the block and B supplies anchors. If only B is AA, B is
the block and A supplies anchors. If both sides are AA, B is the block and each
A column is an anchor. AA–AA in this mode is therefore a column-by-position
block test, not one joint test of every interaction between two entire blocks.

Predictor names must be distinct across the interacting columns and covariates.
This also matters for AA–AA data where different positions can have identically
named encoded residue columns. Selecting different position IDs alone does not
ensure distinct model-column names.

## `interaction()` parameters

```python
results = hp.stats.interaction(
    hladat,
    config,
    a_kind="AA",
    b_kind="DF",
    famfile=None,
    covar=None,
    y=None,
    pair_filter=None,
    cov_block_cols=None,
    baseline_covar_cols=None,
    block_b_df=partners,
    condition_on_a=None,
    verbose=True,
    use_progress_bar=False,
    print_every=500,
)
```

| Parameter | Type | Description |
|---|---|---|
| `hladat` | `HLAdata` | Genotype object containing A and, when applicable, B genotype blocks. |
| `config` | `hp.stats.InteractionConfig` | Model, interaction mode and parallel settings. |
| `a_kind`, `b_kind` | `str` | Required block types; see the block table above. |
| `famfile` | `pandas.DataFrame` or `None` | Optional FAM table with `IID`, and `PHENO` when `y` is omitted. Required for an unindexed phenotype array. |
| `covar` | `pandas.DataFrame`, `str`, `Path` or `None` | IID-indexed covariates, or a CSV path with sample IDs in the first column. Load whitespace-delimited files explicitly with pandas. Required for `b_kind="COV"`. |
| `y` | `pandas.Series`, one-dimensional array-like or `None` | IID-indexed phenotype Series, array aligned to FAM row order, or `None` to use `famfile["PHENO"]`. |
| `pair_filter` | callable or `None` | Function receiving a candidate interaction context and returning `True` to keep or `False` to exclude it. See examples below. |
| `cov_block_cols` | list of strings or `None` | Covariate columns to test as B predictors. Required for `b_kind="COV"`; do not pass for `b_kind="DF"`. |
| `baseline_covar_cols` | list of strings or `None` | Covariate columns always included as adjustment terms. By default, all covariates are used, except the tested B covariates when `b_kind="COV"`. `[]` selects no baseline covariates. |
| `block_b_df` | `pandas.DataFrame` or `None` | Required for `b_kind="DF"`. Rows are sample IIDs and columns are partner predictors. Use numeric dosages or explicitly numeric indicators. |
| `condition_on_a` | `str`, sequence of strings or `None` | Same-kind A-side variant IDs added as adjustment terms. Tasks testing a listed A variant are excluded. |
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

Process workers require importable callbacks and an `if __name__ == "__main__":`
script guard. More workers do not guarantee faster execution.

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

Use `b_kind="COV"` to test AA positions against columns already in your covariate
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
    a_kind="AA",
    b_kind="COV",
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
    hladat, config, a_kind="AA", b_kind="COV",
    covar=covar, y=ra[2],
    cov_block_cols=["SEX", "exposure"],
)
```

This schedules a separate anchor-by-AA-block test for each listed covariate.
It does not jointly test SEX and exposure as one interaction block. All selected
columns are removed from the baseline table, so SEX tests do not automatically
adjust for exposure, or vice versa. Other non-selected covariates remain
baseline adjustment terms by default.

Use `baseline_covar_cols` to name the desired baseline terms explicitly, for
example `["age", "PC1", "PC2"]`. Do not add the tested anchor to that list: its
main effect is already included. To avoid unrelated missing-value exclusions,
pass a covariate table containing the selected interacting predictors and the
adjustment columns you need.

Conditioning with `condition_on_a` works with covariate interactions too; those
genotype columns enter as additional baseline adjustment terms. Covariate B
blocks are identified as `COV_BLOCK` in filter contexts; select a particular
exposure using `anchor_col` in omnibus mode or `b_col` in pairwise mode.

## Selecting interaction tests

`pair_filter` selects tests after the genotype blocks have been built. For
external genotypes, use `read_genotype(..., variants=[...])` first to reduce
reading and memory use; `pair_filter` does not prevent the reader loading data.

For example, select AA genes on A:

```python
pair_filter = hp.stats.filters.only_genes_pair(genes_a=["DRB1", "DQB1"])
filtered = hp.stats.interaction(
    hladat, config, a_kind="AA", b_kind="DF",
    block_b_df=partners, y=phenotype, covar=covar,
    pair_filter=pair_filter,
)
```

A custom omnibus filter can select an AA position and external anchor:

```python
def selected_interaction(ctx):
    return (
        ctx["a_id"] == "AA_DRB1_11_32552129"
        and ctx["anchor_col"] == "rs1"
    )

filtered = hp.stats.interaction(
    hladat, config, a_kind="AA", b_kind="DF",
    block_b_df=partners, y=phenotype, covar=covar,
    pair_filter=selected_interaction,
)
```

Use IDs present in your data. Context keys common to both modes are `analysis`,
`a_kind`, `b_kind`, `a_id`, `b_id`, `a_meta` and `b_meta`. Pairwise mode adds
`a_col` and `b_col`; omnibus mode adds `anchor_col` and `block_cols`. With external
B data, `b_id` is `DF_BLOCK`; select a particular partner using `b_col` or
`anchor_col`, not `b_id`. For covariate B data, `b_id` is `COV_BLOCK`.

## Conditional interaction analysis

```python
conditional = hp.stats.interaction(
    hladat, config, a_kind="AA", b_kind="DF",
    block_b_df=partners, y=phenotype, covar=covar,
    condition_on_a="AA_DRB1_231_32548054",
)
```

Conditioning variant IDs must exist in the A block. Their eligible genotype
columns enter as baseline adjustment terms, and tasks
where the tested A variant is a conditioning variant are excluded. They are not
additional interacting predictors. This option adjusts for A-side variants
only; it does not automatically remove tasks where the same ID occurs on B.
The current interaction runner omits conditioning blocks with no eligible
columns, so a requested empty block does not contribute adjustment terms.



## Saving and plotting

`N_used` reports the final number of complete-case samples for each test.

```python
results.to_csv("aa_genotype_interactions.txt", sep="\t", index=False)
ranked = results.dropna(subset=["LR_p"]).sort_values("LR_p")
```

For pairwise results, sort using `I_p` instead. For AA position-level interaction
plots, see [Visualisation](visualisation.md). Supply the omnibus AA ID column
(`AA_VARIANT`) and the appropriate partner identifiers or genomic coordinates.
External partner IDs such as `rs1` do not themselves provide genomic positions;
supply a position column when an ID cannot be parsed.
