"""P0 smoke tests: the stage-2 foundation is importable and the skeleton hangs together."""


def test_agentproof_foundation_imports():
    from agentproof import AgentState, StateMachine

    assert AgentState is not None
    assert StateMachine is not None


def test_groundproof_package_imports():
    import groundproof

    assert groundproof.__version__ == "0.1.0"


def test_error_family_extends_agentproof():
    from agentproof import AgentProofError

    from groundproof.errors import GroundProofError

    assert issubclass(GroundProofError, AgentProofError)


def test_all_subpackages_import():
    import groundproof.compress
    import groundproof.evals
    import groundproof.grading
    import groundproof.ingest
    import groundproof.retrieval
    import groundproof.steps  # noqa: F401
