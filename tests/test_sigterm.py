def test_sigterm_sets_stop_flag():
    import tinylm.train as T
    T._STOP_REQUESTED = False
    T._handle_sigterm(15, None)  # 15 == SIGTERM
    assert T._should_stop() is True


def test_should_stop_false_by_default():
    import tinylm.train as T
    T._STOP_REQUESTED = False
    assert T._should_stop() is False
