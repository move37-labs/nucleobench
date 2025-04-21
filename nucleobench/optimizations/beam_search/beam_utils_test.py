from nucleobench.optimizations.beam_search import beam_utils

def test_beam_sorter():
    beam_obj = beam_utils.Beam(5)
    beam_obj.put([(2, 'a'), (1, 'b'), (4, 'c'), (3, 'd')])
    assert set(beam_obj.get_items()) == {'a', 'b', 'c', 'd'}
    
    beam_obj.put([(-1, 'e'), (6, 'f'), (5, 'g')])
    assert set(beam_obj.get_items()) == {'e', 'b', 'a', 'd', 'c'}