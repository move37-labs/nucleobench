"""Test some stuff.

To test:
```zsh
pytest nucleobench/common/priority_queue_test.py
```
"""

from nucleobench.common import priority_queue

def test_onesided_priority_queue():
    q = priority_queue.OneSidedPriorityQueue(max_items=4)
    
    def _w(s: str):
        return priority_queue.SearchQItem(
            state=s, 
            fitness=int(s),
            num_edits=0,
        )
    
    for i in range(10):
        q.push(_w(str(i)))
    
    assert len(q.q) == 4
    assert set(q.get(4)) == set(['6', '7', '8', '9'])
    assert set(q.get(3)) == set(['7', '8', '9'])
    
    q.push_batch([_w('-1'), _w('10'), _w('11')])
    assert len(q.q) == 4
    assert set(q.get(4)) == set(['8', '9', '10', '11'])
    
    
def test_twosided_priority_queue():
    q = priority_queue.TwoSidedPriorityQueue(
        max_items=4,
    )
    
    def _w(s: str):
        return priority_queue.SearchQItem(
            state=s, 
            fitness=int(s),
            num_edits=0,
        )
    
    for i in range(10):
        q.push(_w(str(i)))
    
    assert len(q.q) == 4
    assert q.get(4) == ['6', '7', '8', '9']
    
    q.push_batch([_w('-1'), _w('10'), _w('11')])
    assert len(q.q) == 4
    assert q.get(4) == ['8', '9', '10', '11']
    assert q.pop().state == '11'
    assert q.get(4) == ['8', '9', '10']