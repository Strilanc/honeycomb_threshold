import pytest

from measure_tracker import MeasurementTracker, Prev


def test_measurement_tracker_compare_measurement_over_time():
    m = MeasurementTracker()
    with pytest.raises(AssertionError):
        _ = m.measurement_time_set('a')
    with pytest.raises(AssertionError):
        _ = m.measurement_time_set(Prev('a'))

    m.add_measurements('a')
    assert m.measurement_time_set('a') == {0}
    with pytest.raises(AssertionError):
        _ = m.measurement_time_set(Prev('a'))

    m.add_measurements('a')
    assert m.measurement_time_set('a') == {1}
    assert m.measurement_time_set(Prev('a')) == {0}
    assert m.measurement_time_set('a', Prev('a')) == {0, 1}
    assert m.measurement_time_set('a', 'a', Prev('a')) == {0}


def test_measurement_tracker_dummy():
    m = MeasurementTracker()
    m.add_dummies('a')
    m.add_dummies('b', obstacle=True)
    m.add_measurements('c')
    assert m.measurement_time_set('a') == set()
    assert m.measurement_time_set('b') is None
    assert m.measurement_time_set('c') == {0}
    assert m.measurement_time_set('a', 'b') is None
    assert m.measurement_time_set('c', 'b') is None
    assert m.measurement_time_set('a', 'c') == {0}
    assert m.measurement_time_set('a', 'b', 'c') is None
