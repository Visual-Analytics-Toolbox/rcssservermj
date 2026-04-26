from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Enum specifying available soccer game phases."""

    FIRST_HALF = 0
    """The first half of a regular soccer match."""

    SECOND_HALF = 1
    """The second half of a regular soccer match."""

    FIRST_EXTRA_HALF = 2
    """The first half of extra time of a soccer match."""

    SECOND_EXTRA_HALF = 3
    """The second half of extra time of a soccer match."""

    PENALTY_SHOOTING = 4
    """Penalty shootouts."""

    @staticmethod
    def from_value(phase_id: int) -> GamePhase:
        """Fetch the enum entry corresponding to the given game phase ID."""

        for v in GamePhase:
            if v.value == phase_id:
                return v

        logger.warning('Unknown game phase: %i!', phase_id)

        return GamePhase.FIRST_HALF
