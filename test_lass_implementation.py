#!/usr/bin/env python
"""
Quick verification script for LASS implementation
Tests the LASS block components and integration
"""

import torch
import sys
sys.path.insert(0, '/home/bab/Main/mind_cloud_27/UETrack')

from lib.models.uetrack.fastitpn import Natter, S3D, LASS, Block
import torch.nn as nn

def test_natter():
    """Test Natter (Neighborhood Attention) module"""
    print("Testing Natter...")
    natter = Natter(dim=384, num_heads=6, kernel_size=7)
    x = torch.randn(2, 196, 384)  # Batch=2, Tokens=196, Dim=384
    output = natter(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Natter: Input {x.shape} → Output {output.shape}")
    return output

def test_s3d():
    """Test S3D (Spatial-Channel Separation) module"""
    print("Testing S3D...")
    s3d = S3D(dim=384, expansion_ratio=4)
    x = torch.randn(2, 196, 384)
    output = s3d(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ S3D: Input {x.shape} → Output {output.shape}")
    return output

def test_lass():
    """Test LASS block"""
    print("Testing LASS Block...")
    lass = LASS(dim=384, num_heads=6, kernel_size=7, expansion_ratio=4, mlp_ratio=4.)
    x = torch.randn(2, 196, 384)
    output = lass(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ LASS: Input {x.shape} → Output {output.shape}")
    
    # Check that it's learnable
    params = sum(p.numel() for p in lass.parameters())
    print(f"  Total parameters: {params:,}")
    return output

def test_block_with_lass():
    """Test Block with LASS enabled"""
    print("Testing Block with LASS...")
    block = Block(
        dim=384,
        num_heads=6,
        mlp_ratio=4.,
        drop_path=0.1,
        use_lass=True,
    )
    x = torch.randn(2, 196, 384)
    output = block(x, task_index=0)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Block with LASS: Input {x.shape} → Output {output.shape}")
    return output

def test_block_with_mha():
    """Test Block with standard MHA (for comparison)"""
    print("Testing Block with MHA (standard)...")
    block = Block(
        dim=384,
        num_heads=6,
        mlp_ratio=4.,
        drop_path=0.1,
        use_lass=False,
    )
    x = torch.randn(2, 196, 384)
    output = block(x, task_index=0)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Block with MHA: Input {x.shape} → Output {output.shape}")
    return output

def test_gradient_flow():
    """Test backward pass"""
    print("Testing gradient flow...")
    lass = LASS(dim=384, num_heads=6)
    x = torch.randn(2, 196, 384, requires_grad=True)
    output = lass(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradient not computed!"
    print(f"✓ Gradient flow: Loss = {loss.item():.4f}, Grad shape = {x.grad.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("LASS Implementation Verification Tests")
    print("=" * 60)
    
    try:
        test_natter()
        print()
        test_s3d()
        print()
        test_lass()
        print()
        test_block_with_lass()
        print()
        test_block_with_mha()
        print()
        test_gradient_flow()
        
        print()
        print("=" * 60)
        print("✓ All tests passed! LASS implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
