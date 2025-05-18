"""Test ttlcounterset.py

To test:
```zsh
pytest nucleobench/common/ttlcounterset_test.py
```
"""
import pytest

from nucleobench.common.ttlcounterset import TTLCounterSet


def test_fail_read():
    ttl_set = TTLCounterSet(ttl=2)
    assert len(ttl_set) == 0
    
    ttl_set.add("apple")  # Added at counter 0. Expires when counter > 0 + 2 (i.e., at counter = 3)
    ttl_set.add("banana")
    ttl_set.add("orange")
    
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("apple") # This should fail
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("banana") # This should fail
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("orange") # This should fail
    ttl_set.add("orange2")

    ttl_set.increment_counter() # Counter = 1
    
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("apple") # This should fail
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("banana") # This should fail
    with pytest.raises(ValueError) as e_info:
        ttl_set.add("orange") # This should fail
    ttl_set.add("orange3")

    
def test_ttl_counter_set():
    ttl_set = TTLCounterSet(ttl=2)
    assert len(ttl_set) == 0
    
    base_names = ["apple", "banana", "orange"]
    
    ## Step 0
    # Add phase.
    for name in base_names:
        ttl_set.add(f'{name}0')
        
    # Check phase.
    assert len(ttl_set) == 3
    expected_elements = {f"{name}0" for name in base_names}
    assert expected_elements == set(ttl_set.get_current_elements())
    
    ## Step 1
    ttl_set.increment_counter()  # Counter = 1
     # Add phase.
    for name in base_names:
        ttl_set.add(f'{name}1')
        
    # Check phase.
    assert len(ttl_set) == 6
    expected_elements = {f"{name}{s}" for name in base_names for s in range(2)}
    assert expected_elements == set(ttl_set.get_current_elements())
    
    ## Steps 2-10.
    for s in range(2, 11):
        ttl_set.increment_counter()  # Counter = s
        # Add phase.
        for name in base_names:
            ttl_set.add(f'{name}{s}')
            
        # Check phase.
        assert len(ttl_set) == 9
        expected_elements = {f"{name}{_s}" for name in base_names for _s in range(s-2, s+1)}
        assert expected_elements == set(ttl_set.get_current_elements())
        
        
    # Step 11.
    ttl_set.increment_counter()  # Counter = 11
    # No add phase.
        
    # Check phase.
    assert len(ttl_set) == 6
    expected_elements = {f"{name}{_s}" for name in base_names for _s in range(9, 11)}
    assert expected_elements == set(ttl_set.get_current_elements())
    
    # Step 12.
    ttl_set.increment_counter()  # Counter = 12
    # No add phase.
        
    # Check phase.
    assert len(ttl_set) == 3
    expected_elements = {f"{name}10" for name in base_names}
    assert expected_elements == set(ttl_set.get_current_elements())
    
    # Step 13.
    ttl_set.increment_counter()  # Counter = 13
    # No add phase.
        
    # Check phase.
    assert len(ttl_set) == 0
    
    # Check that we can readd expired elements.
    for name in base_names:
        ttl_set.add(f'{name}0')