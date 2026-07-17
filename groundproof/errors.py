"""GroundProof's typed error family, rooted in AgentProof's.

Every error this package raises descends from :class:`GroundProofError`, which itself
descends from ``AgentProofError`` — so callers holding AgentProof machinery can catch
one base class for the whole stack. Concrete errors (retrieval, staleness, ...) are
added in the phase that introduces the failure mode they name.
"""

from agentproof import AgentProofError


class GroundProofError(AgentProofError):
    """Base class for every error GroundProof raises."""


class FetchFailure(GroundProofError):
    """A live corpus fetch failed: network error, bad HTTP status, or malformed payload."""
