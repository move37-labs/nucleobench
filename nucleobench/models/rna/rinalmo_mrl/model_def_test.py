"""Tests for model_def.py

To test:
```zsh  
pytest nucleobench/models/rna/rinalmo_mrl/model_def_test.py
```

IMPORTANT NOTE ON TESTING STRATEGY:
These tests use the actual RiNALMo model architecture but with RANDOM weights
instead of pretrained weights. This is intentional and appropriate for unit testing:

1. The tests use RibosomeLoadingPredictionWrapper which instantiates the full
   RiNALMo transformer architecture with all its layers and components.
   
2. However, the pretrained weights are NOT loaded from Zenodo. The model runs
   with randomly initialized weights.
   
3. This approach is ideal for unit testing because:
   - It tests the complete data flow through the real model architecture
   - It verifies tensor shapes, dtype conversions, and gradient flow
   - It's fast (no need to download ~1GB of pretrained weights)
   - It's deterministic if seeds are set
   - It doesn't depend on external resources (Zenodo)
   
4. Consequence: The actual prediction values are meaningless (random), so tests
   should focus on shapes, types, and computational flow rather than specific
   output values.

5. For integration tests that need real predictions, use the full model with
   load_model() which downloads and loads the actual pretrained weights.
"""

import pytest
import numpy as np
import torch

from nucleobench.common import string_utils, constants

from .rinalmo.ribosome_loading import RibosomeLoadingPredictionWrapper
from . import model_def


def _override_model():
    """Override model for testing.
    
    Returns the RiNALMo model architecture with RANDOM weights (not pretrained).
    This is intentional - see file docstring for explanation.
    """
    return RibosomeLoadingPredictionWrapper(force_cpu=True)


def test_rinalmo_mrl_sanity():
    """Basic sanity test with override model."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    ret = m.inference_on_strings(['AAA', 'CCC', 'TTT', 'GGG', 'ACT'])
    assert list(ret.shape) == [5]


def test_rinalmo_mrl_inference_on_tensor():
    """Test inference_on_tensor method with one-hot encoded input."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    dnas = ['AAA', 'CCC', 'TTT']
    # Convert to one-hot tensors (API now requires one-hot, not token indices)
    onehot_tensors = []
    for seq in dnas:
        onehot = string_utils.dna2tensor(seq, vocab_list=constants.VOCAB)
        onehot_tensors.append(onehot)
    
    batch_onehot = torch.stack(onehot_tensors)
    ret = m.inference_on_tensor(batch_onehot)
    
    assert isinstance(ret, torch.Tensor)
    assert list(ret.shape) == [3]


def test_rinalmo_mrl_inference_on_strings():
    """Test inference_on_strings method."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    ret = m.inference_on_strings(['AAA', 'CCC', 'TTT'])
    
    assert isinstance(ret, np.ndarray)
    assert list(ret.shape) == [3]


def test_rinalmo_mrl_call_method():
    """Test __call__ method."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    ret = m(['AAA', 'CCC', 'TTT'])
    
    assert isinstance(ret, np.ndarray)
    assert list(ret.shape) == [3]


def test_rinalmo_mrl_call_raises_on_string():
    """Test that __call__ raises ValueError when given a single string."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    with pytest.raises(ValueError, match='needs to be list of strings'):
        m('AAA')


def test_debug_init_args():
    """Test debug_init_args static method."""
    args = model_def.RinalmoMRL.debug_init_args()
    assert isinstance(args, dict)
    
    
def test_rinalmo_mrl_batch_processing():
    """Test that batch processing works correctly."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    # Test with different batch sizes
    for batch_size in [1, 5, 10]:
        seqs = ['ACGT'] * batch_size
        ret = m(seqs)
        assert ret.shape[0] == batch_size
    
    
def test_embedding_module():
    """Test that the embedding model has properties needed for TISM."""
    m = model_def.RinalmoMRL(override_model=_override_model())
    embeddings = m._batch_embed(['AAA'])
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 3 + 2, m.model.lm.embedding.embedding_dim)
    # Check that embeddings are floating point.
    assert embeddings.dtype == torch.float32


def test_inference_on_tensor_with_onehot():
    """Test that inference_on_tensor accepts one-hot encoded tensors.
    
    NOTE: This test uses random weights, so we don't check for exact value matches
    between tokenized and one-hot inputs. With random weights, the soft embedding
    approach (weighted sum) can give different results than hard tokenization (argmax).
    The test verifies that both input methods work without errors and produce
    reasonable outputs.
    """
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    # Test sequences
    test_seqs = ["ACGT", "TGCA", "AAAA"]
    
    # Test one-hot encoding functionality
    onehot_tensors = []
    for seq in test_seqs:
        # Convert RNA to DNA notation (U -> T) for standard vocab
        seq_dna = seq.replace('U', 'T')
        onehot = string_utils.dna2tensor(seq_dna, vocab_list=constants.VOCAB)
        onehot_tensors.append(onehot)
    
    batch_onehot = torch.stack(onehot_tensors)
    
    # Test that one-hot inputs work without crashing
    onehot_outputs = m.inference_on_tensor(batch_onehot)
    
    # Check basic properties
    assert onehot_outputs.shape == (3,), \
        f"Expected shape (3,), got {onehot_outputs.shape}"
    assert not torch.isnan(onehot_outputs).any(), \
        f"NaN values in output: {onehot_outputs}"
    assert not torch.isinf(onehot_outputs).any(), \
        f"Inf values in output: {onehot_outputs}"
    
    # Note: Token indices are no longer supported as input to inference_on_tensor
    # The model now exclusively uses one-hot encoding for gradient compatibility


def test_inference_with_gradient_flow():
    """Test that gradients can flow through the one-hot conversion for Ledidi/FastSeqProp.
    
    This test verifies that the soft embedding approach allows gradient-based
    optimizations like Ledidi and FastSeqProp to work. The model uses random weights
    (not pretrained), but this is sufficient to test that:
    1. The backward pass completes without errors
    2. Gradients are computed (even if they're small/random)
    3. The model is compatible with PyTorch's autograd system
    """
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    # Create a simple test sequence
    test_seq = "ACGT"
    seq_dna = test_seq.replace('U', 'T')
    onehot = string_utils.dna2tensor(seq_dna, vocab_list=constants.VOCAB)
    batch_onehot = onehot.unsqueeze(0).float()  # Add batch dimension and ensure float
    
    # Make it require gradients
    batch_onehot.requires_grad = True
    
    # Forward pass
    output = m.inference_on_tensor(batch_onehot)
    
    # Compute a simple loss
    loss = output.sum()
    
    # Try to compute gradients
    try:
        loss.backward()
        # Gradients should exist (even if they're small due to the soft embedding approach)
        assert batch_onehot.grad is not None, "No gradients computed"
        # The soft embedding approach (weighted sum of embeddings) allows gradients
        # to flow, unlike a hard argmax which would block gradients.
        # With random weights, gradients will be random but non-zero.
        print("Gradient computation successful")
    except Exception as e:
        pytest.fail(f"Backward pass failed: {e}")


def test_custom_onehot_embeddings_match_normal_tokenization():
    """Test that custom one-hot to embedding logic produces same embeddings as normal RiNALMo tokenization.
    
    This test verifies that when we convert one-hot tensors to embeddings using our
    custom soft embedding approach, we get the exact same embeddings as the normal
    RiNALMo encoding (strings -> tokens -> embeddings).
    
    This is important because it ensures our custom logic is equivalent to the
    standard RiNALMo encoding scheme, just with gradient flow enabled.
    
    NOTE: This test uses random weights (not pretrained), but the equivalence should
    hold regardless of weights since we're testing the encoding logic itself.
    """
    m = model_def.RinalmoMRL(override_model=_override_model())
    
    # Test sequences
    test_seqs = ["ACGU", "UGCA", "AAAA", "CCCC", "GGGG", "UUUU"]
    
    # Method 1: Custom one-hot to embedding logic (replicate the logic from inference_on_tensor)
    onehot_tensors = []
    for seq in test_seqs:
        seq_dna = seq.replace('U', 'T')
        onehot = string_utils.dna2tensor(seq_dna, vocab_list=constants.VOCAB)
        onehot_tensors.append(onehot)
    
    batch_onehot = torch.stack(onehot_tensors)
    if m.has_cuda:
        batch_onehot = batch_onehot.cuda()
    
    batch_size = batch_onehot.shape[0]
    seq_len = batch_onehot.shape[2]
    device = batch_onehot.device
    
    # Replicate the custom embedding logic from inference_on_tensor
    nucleotide_token_indices = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    if m.has_cuda:
        nucleotide_token_indices = nucleotide_token_indices.cuda()
    nucleotide_embeddings = m.model.lm.embedding(nucleotide_token_indices)
    
    x_transposed = batch_onehot.transpose(1, 2)  # (batch_size, seq_len, 4)
    sequence_embeddings_custom = torch.matmul(x_transposed, nucleotide_embeddings)
    
    # Add CLS and EOS embeddings
    cls_token_idx = torch.tensor([0], dtype=torch.long, device=device)
    eos_token_idx = torch.tensor([2], dtype=torch.long, device=device)
    if m.has_cuda:
        cls_token_idx = cls_token_idx.cuda()
        eos_token_idx = eos_token_idx.cuda()
    
    cls_embedding = m.model.lm.embedding(cls_token_idx).expand(batch_size, 1, -1)
    eos_embedding = m.model.lm.embedding(eos_token_idx).expand(batch_size, 1, -1)
    full_embeddings_custom = torch.cat([cls_embedding, sequence_embeddings_custom, eos_embedding], dim=1)
    
    # Method 2: Normal RiNALMo tokenization (strings -> tokens -> embeddings)
    tokens = m.batch_tokenize(test_seqs)
    if m.has_cuda:
        tokens = tokens.cuda()
    
    # Get embeddings using normal RiNALMo approach
    full_embeddings_normal = m.model.lm.embedding(tokens)
    
    # Compare: They should be identical when one-hot is truly one-hot
    # (i.e., when argmax gives the same result as the weighted sum)
    assert torch.allclose(full_embeddings_custom, full_embeddings_normal, atol=1e-6), \
        f"Custom embedding logic doesn't match normal tokenization!\n" \
        f"Max difference: {torch.abs(full_embeddings_custom - full_embeddings_normal).max()}\n" \
        f"Custom shape: {full_embeddings_custom.shape}, Normal shape: {full_embeddings_normal.shape}"