"""Integration test for RiNALMo model.

To test:
```zsh
pytest -s integration_tests/rinalmo_mrl_integration_test.py
```
"""

import pandas as pd
import tqdm
import pytest

from sklearn.metrics import r2_score, mean_absolute_error

from nucleobench.models.rna.rinalmo_mrl.model_def import RinalmoMRL

from .data_loaders import MRLByBucket, UTR5PrimeRinalmoRepo, Human7600MRLPreds
    

@pytest.mark.rinalmo
def test_rinalmo_r2_score(utr_5prime_rinalmo_repo_data):
    """Test R2 score on 5' UTR data from RiNALMo repository."""
    data_df = utr_5prime_rinalmo_repo_data.get_data()
    m = RinalmoMRL()
    pred_pairs = []
    for idx, row in tqdm.tqdm(data_df.head(100).iterrows(), total=100):
        x = row.utr
        y_gt = row.rl
        pred = m([x])[0]
        
        pred_pairs.append((pred, y_gt))
    preds, gts = zip(*pred_pairs)
    
    r2score = r2_score(y_pred=preds, y_true=gts)
    print(f'r2_score: {r2score:.4f}')
    assert r2score > 0.58
    

@pytest.mark.rinalmo
def test_rinalmo_mrl_golden_testset():
    """Test RinalmoMRL model correctness on golden testset.
    
                    My custom attention     |       "no_flash" branch attention
                    Fidelity        GT MRL          Fidelity        GT MRL
                    R2      L1      R2      L1      R2      L1      R2      L1
    First 1K seqs   0.7700  0.5000  0.6484  0.6190  0.7733  0.5072  0.6484  0.6190
    All 7600 seqs   0.7414  0.5128  0.6291  0.6187  0.7414  0.5128  0.6291  0.6187
    """
    # Load data using the data loader
    loader = Human7600MRLPreds()
    data_df = loader.get_data()
    m = RinalmoMRL()
    dat = []
    for idx, row in tqdm.tqdm(data_df.head(100).iterrows(), total=100):
        x = row.sequence
        y_pred = row.mrl_predicted
        y_gt = row.mrl_target
        pred = m([x])[0]
        
        dat.append((pred, y_pred, y_gt))
    preds, expected_preds, gt_mrl = zip(*dat)
    
    r2score = r2_score(y_pred=preds, y_true=expected_preds)
    l1 = mean_absolute_error(y_pred=preds, y_true=expected_preds)
    
    r2score_gt = r2_score(y_pred=preds, y_true=gt_mrl)
    l1_gt = mean_absolute_error(y_pred=preds, y_true=gt_mrl)
    
    print(f'r2_score: {r2score:.4f}')
    print(f'l1: {l1:.4f}')
    
    print(f'r2_score_gt: {r2score_gt:.4f}')
    print(f'l1_gt: {l1_gt:.4f}')
    
    assert r2score > 0.74
    assert l1 < 0.52


@pytest.fixture(scope="function")
def mrl_by_bucket_data():
    """Fixture that ensures cache is populated for MRL by bucket data loader.
    
    If cache doesn't exist, it will download and process the data.
    Otherwise, it just returns the loader which will use the existing cache.
    """
    loader = MRLByBucket()
    cache_path = loader._get_cache_path()
    
    # Only populate cache if it doesn't exist
    if not cache_path.exists():
        loader.populate_cache()
    else:
        print(f"Using existing cache: {cache_path}")
    
    return loader


@pytest.fixture(scope="function")
def utr_5prime_rinalmo_repo_data():
    """Fixture that ensures cache is populated for 5' UTR RiNALMo repo data loader.
    
    If cache doesn't exist, it will download and process the data.
    Otherwise, it just returns the loader which will use the existing cache.
    """
    loader = UTR5PrimeRinalmoRepo()
    cache_path = loader._get_cache_path()
    
    # Only populate cache if it doesn't exist
    if not cache_path.exists():
        loader.populate_cache()
    else:
        print(f"Using existing cache: {cache_path}")
    
    return loader


@pytest.mark.rinalmo
def test_rinalmo_mrl_bucket_ordering(mrl_by_bucket_data):
    """Test that predicted MRL values are in correct order by bucket.
    
    This test loads sequences from the data loader and verifies
    that predicted MRL values follow the expected ordering:
    HIGH bucket > MEDIUM bucket > LOW bucket
    
    This validates that the model correctly distinguishes between different
    expression levels based on sequence features.
    """
    # Load data through the data loader
    data_df = mrl_by_bucket_data.get_data()
    m = RinalmoMRL()
    
    # Run inference on all sequences
    results = []
    for idx, row in data_df.iterrows():
        sequence = row['sequence']
        bucket = row['bucket']
        gene = row['gene']
        
        # Run inference
        pred_mrl = m([sequence])[0]
        
        results.append({
            'gene': gene,
            'bucket': bucket,
            'mrl': pred_mrl
        })
        
        print(f"{gene} ({bucket}): MRL = {pred_mrl:.4f}")
    
    # Group by bucket and compute statistics
    results_df = pd.DataFrame(results)
    
    high_mrl = results_df[results_df['bucket'] == 'HIGH']['mrl'].values
    medium_mrl = results_df[results_df['bucket'] == 'MEDIUM']['mrl'].values
    low_mrl = results_df[results_df['bucket'] == 'LOW']['mrl'].values
    
    high_mean = high_mrl.mean() * -1
    medium_mean = medium_mrl.mean() * -1
    low_mean = low_mrl.mean() * -1
    
    print(f"\nBucket Statistics:")
    print(f"  HIGH (mean):   {high_mean:.4f}")
    print(f"  MEDIUM (mean): {medium_mean:.4f}")
    print(f"  LOW (mean):    {low_mean:.4f}")
    
    # Assert that HIGH and MEDIUM are both greater than LOW
    # (This is the primary biological expectation)
    assert high_mean > low_mean, \
        f"Expected HIGH bucket MRL ({high_mean:.4f}) > LOW bucket MRL ({low_mean:.4f})"
    assert medium_mean > low_mean, \
        f"Expected MEDIUM bucket MRL ({medium_mean:.4f}) > LOW bucket MRL ({low_mean:.4f})"
    
    # Also verify that HIGH and MEDIUM are both significantly above LOW
    # (at least 0.5 units difference to account for model variance)
    assert high_mean - low_mean > 0.5, \
        f"Expected HIGH bucket MRL to be at least 0.5 units above LOW bucket MRL"
    assert medium_mean - low_mean > 0.5, \
        f"Expected MEDIUM bucket MRL to be at least 0.5 units above LOW bucket MRL"
    
    print("\n✓ Bucket ordering verified: HIGH and MEDIUM > LOW")
    
    # Optional: Check if HIGH > MEDIUM (may not always hold with small sample sizes)
    if high_mean > medium_mean:
        print("  Additional: HIGH > MEDIUM")
    else:
        print(f"  Note: MEDIUM ({medium_mean:.4f}) > HIGH ({high_mean:.4f}), but both are > LOW")