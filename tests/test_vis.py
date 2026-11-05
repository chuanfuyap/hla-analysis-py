from decimal import Decimal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pytest

import hapy as hp
from hapy.vis._utils import (parse_aa_identifier, parse_partner_identifier, format_pvalue,
                             prepare_data, select_hits)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def association():
    return pd.DataFrame({'VARIANT':['AA_B_63_31324057','AA_DPB1_9_33043703','AA_DPB1_35_33043625',
                                    'AA_DRB1_11_32552129','AA_B_11_31324200'],
                         'L':[5.38,5.14,5.37,412-np.log10(6.09),0.]})


@pytest.fixture
def interactions(association):
    return association.rename(columns={'VARIANT':'AA_VARIANT'}).assign(
        TCR_POS=[38175678,37289819,37289819,39000000,38175678],
        PARTNER=['chr7:38175678','chr7:37289819','chr7:37289819','chr7:39000000','chr7:38175678'])


def test_exact_aa_position_parse():
    assert parse_aa_identifier('AA_DRB1_11_32552129')==('DRB1',11,32552129)
    assert parse_aa_identifier('AA_dpb1_-5_33000000')==('DPB1',-5,33000000)


@pytest.mark.parametrize('identifier',['AA_DRB1_11_32552129_A','AA_DQB1_13_32632813_exon2',
                                       'SNP_A_11_32552129','AA_DRB1_11','AA_DRB1_x_32552129',
                                       'AA_DRB1_11_0'])
def test_residue_or_malformed_identifier_rejected(identifier):
    with pytest.raises(ValueError,match='position identifier'):
        parse_aa_identifier(identifier)


@pytest.mark.parametrize('identifier',['chr7:38175678','7:38175678:A:G','chr7_38175678_A_G','SNPS_A_7_38175678_exon1'])
def test_partner_parse(identifier):
    assert parse_partner_identifier(identifier)==('7',38175678)


def test_pvalue_format_extreme():
    assert format_pvalue(412-np.log10(6.09))=='6.09 × 10⁻⁴¹²'
    assert format_pvalue(6-np.log10(4.17))=='4.17 × 10⁻⁶'
    assert format_pvalue(0)=='1'
    assert format_pvalue(3-np.log10(9.9999))=='0.01'


def test_raw_decimal_and_string_underflow():
    frame=pd.DataFrame({'VARIANT':['AA_B_63_31324057','AA_DPB1_9_33043703'],
                        'P':['6.09e-412',Decimal('4.17e-6')]})
    _,ax=hp.vis.plot_aa_association(frame,pval_col='P',highlight_top=True,colorbar=False, x_axis='genomic')
    assert '6.09 × 10⁻⁴¹²' in ax.texts[0].get_text()


def test_coordinates_highlights_gene_specific_and_no_mutation(association):
    before=association.copy(deep=True)
    fig,ax=hp.vis.plot_aa_association(association,log10p_col='L',highlights=[('B',11)],colorbar=False, x_axis='genomic')
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,0],[31324057,33043703,33043625,32552129,31324200])
    assert len(ax.texts)==1 and 'B amino acid position 11' in ax.texts[0].get_text()
    assert ax.collections[1].get_offsets()[0,0]==31324200
    pd.testing.assert_frame_equal(before,association)
    assert fig is ax.figure


def test_topn_gene_filter_and_manual_highlight(association):
    _,ax=hp.vis.plot_aa_association(association,log10p_col='L',genes=['B','DPB1'],
                                   highlight_top=2,highlights=['AA_B_11_31324200'],colorbar=False, x_axis='genomic')
    assert len(ax.collections[0].get_offsets())==4 and len(ax.texts)==3
    assert all('DRB1' not in text.get_text() for text in ax.texts)


def test_no_annotations_offsets_custom_and_warning(association):
    with pytest.warns(UserWarning,match='not found'):
        _,ax=hp.vis.plot_aa_association(association,log10p_col='L',highlights=[('B',99)],colorbar=False, x_axis='genomic')
    assert not ax.texts
    _,ax=hp.vis.plot_aa_association(association,log10p_col='L',highlight_top=True,annotate_p=False, x_axis='genomic')
    assert len(ax.collections)==2 and not ax.texts
    _,ax=hp.vis.plot_aa_association(association,log10p_col='L',highlights=[('B',63)],annotate_p=False,
        annotations={('B',63):'Custom hit'},annotation_offsets={('B',63):(30,-20)}, x_axis='genomic')
    assert ax.texts[0].get_text()=='Custom hit' and ax.texts[0].get_position()==(30,-20)


@pytest.mark.parametrize('kind',['scatter3d','heatmap'])
def test_interaction_mapping_labels_and_selection(interactions,kind):
    before=interactions.copy(deep=True)
    _,ax=hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',partner_variant_col='PARTNER',
        log10p_col='L',kind=kind,highlights=[('B',63,38175678),('DPB1',9,37289819)],colorbar=False, x_axis='genomic')
    assert len(ax.texts)==2
    assert 'chr7:38175678' in ax.texts[0].get_text() and 'p =' in ax.texts[0].get_text()
    coordinates=ax.collections[0]._offsets3d if kind=='scatter3d' else ax.collections[0].get_offsets().T
    np.testing.assert_array_equal(coordinates[0],[31324057,33043703,33043625,32552129,31324200])
    np.testing.assert_array_equal(coordinates[1],interactions.TCR_POS)
    if kind=='scatter3d':
        ax.view_init(elev=45,azim=20)
    ax.figure.canvas.draw()
    pd.testing.assert_frame_equal(before,interactions)


def test_sparse_heatmap_is_observed_points_only(interactions):
    _,ax=hp.vis.plot_aa_interaction(interactions,partner_variant_col='PARTNER',log10p_col='L',
                                   kind='heatmap',annotate_cells=True,missing_color='pink', x_axis='genomic')
    assert len(ax.collections[0].get_offsets())==5 # not 5 by 3 fabricated cells
    assert len(ax.texts)==5
    assert ax.collections[0].get_array()[-1]==0 # observed non-significant P=1 remains visible
    assert ax.get_facecolor()==matplotlib.colors.to_rgba('pink')


def test_duplicate_positions_not_combined(association):
    duplicate=pd.concat([association,association.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='position-level'):
        hp.vis.plot_aa_association(duplicate,log10p_col='L', x_axis='genomic')


def test_distinct_coordinates_not_merged(association):
    extra=association.iloc[:1].copy()
    extra.VARIANT='AA_B_63_31324058'
    _,ax=hp.vis.plot_aa_association(pd.concat([association,extra]),log10p_col='L',colorbar=False, x_axis='genomic')
    assert len(ax.collections[0].get_offsets())==6


@pytest.mark.parametrize('kind',['scatter3d','heatmap'])
def test_duplicate_interaction_cells_explicit_only(interactions,kind):
    dup=interactions.iloc[:1].copy();dup.L=20
    frame=pd.concat([interactions,dup],ignore_index=True)
    with pytest.raises(ValueError,match='coordinate'):
        hp.vis.plot_aa_interaction(frame,partner_pos_col='TCR_POS',log10p_col='L',kind=kind,duplicate='error', x_axis='genomic')
    with pytest.warns(UserWarning,match='visual aggregation'):
        _,ax=hp.vis.plot_aa_interaction(frame,partner_pos_col='TCR_POS',log10p_col='L',kind=kind,duplicate='max', x_axis='genomic')
    assert len(ax.collections[0].get_array())==5 and 20 in ax.collections[0].get_array()


def test_same_coordinates_different_biological_ids_raise(interactions):
    other=interactions.iloc[:1].copy();other.AA_VARIANT='AA_C_63_31324057'
    with pytest.raises(ValueError,match='coordinate'):
        hp.vis.plot_aa_interaction(pd.concat([interactions,other]),partner_pos_col='TCR_POS',log10p_col='L',duplicate='error', x_axis='genomic')


@pytest.mark.parametrize('value',[0,-.1,1.1,np.inf,'broken'])
def test_invalid_probabilities_rejected(association,value):
    frame=association.assign(P=.1);frame.loc[0,'P']=value
    with pytest.raises(ValueError):hp.vis.plot_aa_association(frame,pval_col='P', x_axis='genomic')


def test_missing_and_nonfinite_log_values(association):
    frame=association.copy();frame.loc[0,'L']=np.nan;frame.loc[1,'L']=np.inf
    with pytest.warns(UserWarning):
        _,ax=hp.vis.plot_aa_association(frame,log10p_col='L',colorbar=False, x_axis='genomic')
    assert len(ax.collections[0].get_offsets())==3
    frame.L=-1
    with pytest.raises(ValueError):hp.vis.plot_aa_association(frame,log10p_col='L', x_axis='genomic')
    with pytest.raises(ValueError,match='No valid'):
        hp.vis.plot_aa_association(association,log10p_col='L',genes='C', x_axis='genomic')


def test_partner_validation(interactions):
    with pytest.raises(ValueError,match='disagree'):
        hp.vis.plot_aa_interaction(interactions.assign(TCR_POS=1),partner_pos_col='TCR_POS',partner_variant_col='PARTNER',log10p_col='L', x_axis='genomic')
    frame=interactions.copy();frame.loc[0,'PARTNER']='chr8:38175678'
    with pytest.raises(ValueError,match='multiple chromosomes'):
        hp.vis.plot_aa_interaction(frame,partner_variant_col='PARTNER',log10p_col='L', x_axis='genomic')
    frame=interactions.copy();frame.loc[0,'PARTNER']='rs123'
    with pytest.raises(ValueError,match='supply partner_pos_col'):
        hp.vis.plot_aa_interaction(frame,partner_variant_col='PARTNER',log10p_col='L', x_axis='genomic')
    _,ax=hp.vis.plot_aa_interaction(frame,partner_variant_col='PARTNER',partner_pos_col='TCR_POS',log10p_col='L', x_axis='genomic')
    assert ax is not None


def test_shared_scale_axis_format_thresholds_existing_axes(association,interactions):
    norm=Normalize(0,500)
    fig,ax=plt.subplots()
    out=hp.vis.plot_aa_association(association,log10p_col='L',norm=norm,ax=ax,position_unit='Mb',
        reference_build='GRCh38',threshold_log10p=[5,10],colorbar=False,x_ticks=4,title='Title', x_axis='genomic')
    assert out==(fig,ax) and ax.collections[0].norm is norm
    assert ax.xaxis.get_major_formatter()(32552129,0)=='32.5521'
    assert 'GRCh38' in ax.get_xlabel() and len(ax.lines)==2
    _,ay=hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',log10p_col='L',kind='heatmap',norm=norm, x_axis='genomic')
    assert ay.collections[0].norm is norm
    with pytest.raises(ValueError,match='not both'):
        hp.vis.plot_aa_association(association,log10p_col='L',norm=norm,vmin=1, x_axis='genomic')
    with pytest.raises(ValueError,match='3D axes'):
        hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',log10p_col='L',ax=ax, x_axis='genomic')


@pytest.mark.parametrize('extension',['png','pdf','svg'])
@pytest.mark.parametrize('kind',['association','scatter3d','heatmap'])
def test_headless_exports(association,interactions,tmp_path,extension,kind):
    path=tmp_path/f'plot.{extension}'
    common=dict(x_axis='genomic',log10p_col='L',save_path=path,show=False,highlight_top=True,dpi=90)
    if kind=='association':fig,ax=hp.vis.plot_aa_association(association,**common)
    else:fig,ax=hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',kind=kind,**common)
    assert path.stat().st_size>1000
    assert ax.figure is fig


def test_public_api_no_qqplot():
    assert hp.plot_aa_association is hp.vis.plot_aa_association
    assert hp.plot_aa_interaction is hp.vis.plot_aa_interaction
    assert not hasattr(hp.vis,'qqplot')


def test_hapy_meta_interaction_key(interactions):
    frame=interactions.assign(INTERACTION=interactions.AA_VARIANT+'*'+interactions.PARTNER)
    _,ax=hp.vis.plot_aa_interaction(frame,variant_col='INTERACTION',partner_variant_col='INTERACTION',
                                   log10p_col='L',kind='heatmap',highlights=[('B',63,38175678)], x_axis='genomic')
    assert 'B amino acid position 63' in ax.texts[0].get_text()
    assert 'chr7:38175678' in ax.texts[0].get_text()
    assert len(ax.collections[0].get_offsets())==5


def test_partner_missing_and_invalid_highlights(interactions):
    frame=interactions.copy();frame.loc[0,'TCR_POS']=np.nan
    with pytest.warns(UserWarning,match='partner coordinates'):
        _,ax=hp.vis.plot_aa_interaction(frame,partner_pos_col='TCR_POS',log10p_col='L',kind='heatmap', x_axis='genomic')
    assert len(ax.collections[0].get_offsets())==4
    with pytest.raises(ValueError,match='Highlight must'):
        hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',log10p_col='L',highlights=[('B',63)], x_axis='genomic')
    with pytest.raises(ValueError,match='not both'):
        hp.vis.plot_aa_association(interactions,variant_col='AA_VARIANT',pval_col='L',log10p_col='L', x_axis='genomic')


def test_identifier_only_anchor_column_user_example():
    frame=pd.DataFrame({'AA_VARIANT':['AA_DRB1_11_32552129'],
                        'Anchor_col':['chr7_37338247_T_C'], 'LR_p':[0.001]})
    for kind in ['scatter3d','heatmap']:
        fig,ax=hp.vis.plot_aa_interaction(frame,partner_variant_col='Anchor_col',
                                         pval_col='LR_p',kind=kind,colorbar=False, x_axis='genomic')
        coordinates=ax.collections[0]._offsets3d if kind=='scatter3d' else ax.collections[0].get_offsets().T
        assert coordinates[1][0]==37338247
        assert ax.collections[0].get_array()[0]==pytest.approx(3.)
        fig.canvas.draw()
        plt.close(fig)


def test_numeric_probability_fast_path_matches_decimal(association,monkeypatch):
    import hapy.vis._utils as utils
    probabilities=np.array([1.,.05,1e-100,np.nextafter(0.,1.),.125])
    frame=association.assign(P=probabilities)
    reference=utils.prepare_data(frame.assign(P=[Decimal.from_float(float(p)) for p in probabilities]),
                                 'VARIANT','P',None,None)
    def unexpected_decimal(value):
        raise AssertionError('Numeric inputs should not call Decimal conversion')
    monkeypatch.setattr(utils,'_log_probability',unexpected_decimal)
    result=utils.prepare_data(frame,'VARIANT','P',None,None)
    np.testing.assert_allclose(result.score,reference.score,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('kind', ['scatter3d', 'heatmap'])
def test_distinct_partner_variants_same_coordinate_preserved(kind):
    frame=pd.DataFrame({'AA_VARIANT':['AA_A_-15_29910359']*2,
                        'Anchor_col':['7_37438406_T_A_T','7_37438406_T_G_T'],
                        'LR_p':[.087101,.696269]})
    kwargs=dict(x_axis='genomic',partner_variant_col='Anchor_col',pval_col='LR_p',kind=kind,colorbar=False)
    if kind=='heatmap':
        with pytest.warns(UserWarning,match='all observations are retained'):
            fig,ax=hp.vis.plot_aa_interaction(frame,**kwargs)
    else:
        fig,ax=hp.vis.plot_aa_interaction(frame,**kwargs)
        np.testing.assert_allclose(ax.collections[0]._offsets3d[2],-np.log10(frame.LR_p))
    assert len(ax.collections[0].get_array())==2
    np.testing.assert_allclose(ax.collections[0].get_array(),-np.log10(frame.LR_p))
    fig.canvas.draw()


@pytest.mark.parametrize('kind',['association','heatmap','scatter3d'])
def test_amino_acid_axis(kind,association,interactions):
    opts=dict(log10p_col='L',genes='DPB1',x_axis='amino_acid',position_unit='Mb',colorbar=False,highlight_top=True)
    if kind=='association':fig,ax=hp.vis.plot_aa_association(association,**opts)
    else:fig,ax=hp.vis.plot_aa_interaction(interactions,partner_variant_col='PARTNER',kind=kind,**opts)
    coordinates=ax.collections[0]._offsets3d if kind=='scatter3d' else ax.collections[0].get_offsets().T
    np.testing.assert_array_equal(coordinates[0],[9,35])
    assert 'HLA-DPB1 amino acid sequence position'==ax.get_xlabel()
    assert ax.xaxis.get_major_formatter()(35,0)=='35'
    if kind!='association':
        np.testing.assert_array_equal(coordinates[1],[37289819,37289819])
        assert 'Mb' in ax.get_ylabel()
    fig.canvas.draw()


def test_amino_acid_axis_requires_gene(association,interactions):
    with pytest.raises(ValueError,match='single HLA gene'):
        hp.vis.plot_aa_association(association,log10p_col='L',x_axis='amino_acid')
    with pytest.raises(ValueError,match='single HLA gene'):
        hp.vis.plot_aa_interaction(interactions,partner_pos_col='TCR_POS',log10p_col='L',x_axis='amino_acid')
    with pytest.raises(ValueError,match='x_axis'):
        hp.vis.plot_aa_association(association,log10p_col='L',x_axis='bad')


@pytest.mark.parametrize('kind',['heatmap','scatter3d'])
def test_compact_callouts_and_coloured_highlights(interactions,kind):
    fig,ax=hp.vis.plot_aa_interaction(
        interactions,partner_variant_col='PARTNER',log10p_col='L',kind=kind,
        highlights=[('B',63,38175678)],annotation_style='compact',annotation_partner=False,
        highlight_color='gold',highlight_edgecolor='red',highlight_linewidth=2,
        marker_size=2,alpha=.2,colorbar=False, x_axis='genomic')
    fig.canvas.draw()
    label=ax.texts[0]
    assert label.get_text().startswith('B pos 63\np = ')
    assert 'chr7' not in label.get_text()
    if kind=='scatter3d':
        marker=next(line for line in ax.lines if hasattr(line,'points'))
        assert marker.get_markerfacecolor()=='gold'
        assert marker.get_markeredgecolor()=='red'
        assert marker.get_alpha()==1.
        assert marker.get_markersize()**2>=80
        assert marker.get_zorder()>ax.collections[0].get_zorder()
        assert label.get_zorder()>marker.get_zorder()
        previous=label.xy
        ax.view_init(elev=40,azim=20)
        fig.canvas.draw()
        assert label.xy!=previous
    else:
        np.testing.assert_allclose(ax.collections[1].get_facecolors()[0],matplotlib.colors.to_rgba('gold'))
        assert ax.collections[1].get_alpha()==1.


def test_auto_labels_avoid_overlaps_and_export_after_rotation(tmp_path):
    frame=pd.DataFrame({'AA_VARIANT':['AA_B_11_31324000','AA_B_12_31324001','AA_B_13_31324002'],
                        'PARTNER':['chr7:38000000']*3,'L':[5.,5.,5.]})
    fig,ax=hp.vis.plot_aa_interaction(frame,partner_variant_col='PARTNER',log10p_col='L',
        highlight_top=3,annotation_style='compact',annotation_partner=False,
        annotation_fontsize=10,colorbar=False, x_axis='genomic')
    for azim in [-60,30]:
        ax.view_init(azim=azim)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        from matplotlib.text import Text
        boxes=[Text.get_window_extent(text,renderer) for text in ax.texts]
        assert not any(boxes[i].overlaps(boxes[j]) for i in range(3) for j in range(i))
    fig.savefig(tmp_path/'rotated.svg')
    assert (tmp_path/'rotated.svg').stat().st_size>1000


def test_manual_annotation_offsets_stay_fixed(interactions):
    fig,ax=hp.vis.plot_aa_interaction(interactions,partner_variant_col='PARTNER',log10p_col='L',
        highlights=[('B',63,38175678)],annotation_offsets={('B',63,38175678):(20,-15)}, x_axis='genomic')
    fig.canvas.draw()
    assert ax.texts[0].get_position()==(20,-15)


@pytest.mark.parametrize('identifier,expected',[
    ('7:38175678','chr7:38175678'),
    ('CHR7:38175678','chr7:38175678'),
    ('chr7_38175678','chr7:38175678'),
    ('chr7_38175678_T_C','chr7_38175678_T_C'),
    ('7:38175678:T:C','7:38175678:T:C'),
    ('rs123','rs123 (chr7:38175678)'),
])
def test_partner_labels_do_not_repeat_coordinates(identifier,expected):
    from hapy.vis._utils import add_partners
    data=pd.DataFrame({'VARIANT':['AA_B_63_31324057'],'P':[.01],
                       'PARTNER':[identifier],'POS':[38175678]})
    prepared=prepare_data(data,'VARIANT','P',None,None)
    result=add_partners(prepared,data,'POS','PARTNER','CHR7')
    assert result.partner_label.iloc[0]==expected


@pytest.mark.parametrize('kind',['association','heatmap','scatter3d'])
def test_default_axis_is_sequence_position(kind,association,interactions):
    kwargs=dict(log10p_col='L',genes='DPB1',colorbar=False)
    if kind=='association':fig,ax=hp.vis.plot_aa_association(association,**kwargs)
    else:fig,ax=hp.vis.plot_aa_interaction(interactions,partner_variant_col='PARTNER',kind=kind,**kwargs)
    coords=ax.collections[0]._offsets3d if kind=='scatter3d' else ax.collections[0].get_offsets().T
    np.testing.assert_array_equal(coords[0],[9,35])
    assert ax.get_xlabel()=='HLA-DPB1 amino acid sequence position'
    fig.canvas.draw()


def test_default_sequence_axis_requires_single_gene(association):
    with pytest.raises(ValueError,match='single HLA gene'):
        hp.vis.plot_aa_association(association,log10p_col='L')


@pytest.mark.parametrize('kind',['association','heatmap','scatter3d'])
def test_equally_spaced_axis_and_info(kind,association,interactions,capsys):
    opts=dict(x_axis='equally_spaced',log10p_col='L',colorbar=False,highlight_top=True,position_unit='Mb')
    if kind=='association':fig,ax=hp.vis.plot_aa_association(association,**opts)
    else:fig,ax=hp.vis.plot_aa_interaction(interactions,partner_variant_col='PARTNER',kind=kind,**opts)
    coords=ax.collections[0]._offsets3d if kind=='scatter3d' else ax.collections[0].get_offsets().T
    np.testing.assert_array_equal(coords[0],[0,4,3,2,1])
    assert ax.get_xlabel()=='HLA amino acid positions (equally spaced)'
    assert 'INFO:' in capsys.readouterr().out
    assert ax.xaxis.get_major_formatter()(0,0)=='HLA-B:63'
    assert ax.xaxis.get_major_formatter()(4,0)=='HLA-DPB1:9'
    fig.canvas.draw()


def test_equally_spaced_order_independent_and_distinct_positions(association,capsys):
    extra=association.iloc[:1].copy();extra.VARIANT='AA_B_63_31324058'
    data=pd.concat([association,extra],ignore_index=True).sample(frac=1,random_state=7)
    _,ax=hp.vis.plot_aa_association(data,log10p_col='L',x_axis='equally_spaced',colorbar=False)
    assert sorted(ax.collections[0].get_offsets()[:,0])==list(range(6))
    assert 'genomic-coordinate order' in capsys.readouterr().out


def test_numeric_axes_do_not_print_spacing_info(association,capsys):
    hp.vis.plot_aa_association(association,log10p_col='L',genes='B')
    assert 'INFO:' not in capsys.readouterr().out
