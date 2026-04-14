from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from rcsssmj.games.soccer.soccer_fields import SoccerFieldVersions

logger = logging.getLogger(__name__)


class SoccerRuleBooks(Enum):
    """Enum specifying available soccer rule books."""

    UNKNOWN = 'unknown'
    """Unknown rule book (only used to indicate an error)."""

    FIFA = 'fifa'
    """Official FIFA rule book."""

    SSIM = 'ssim'
    """Latest version of the RoboCup Soccer Simulation League competition rule book."""

    SSIM_26 = 'ssim26'
    """2026 version of the RoboCup Soccer Simulation League competition rule book."""

    @staticmethod
    def from_value(name: str) -> SoccerRuleBooks:
        """Fetch the enum entry corresponding to the given rule book name."""

        for v in SoccerRuleBooks:
            if v.value == name:
                return v

        logger.warning('Unknown rule book: %s!', name)

        return SoccerRuleBooks.UNKNOWN


def create_soccer_rule_book(name: str) -> SoccerRules:
    """Create the soccer rule book for the given name."""

    name_id = SoccerRuleBooks.from_value(name)

    if name_id in (SoccerRuleBooks.SSIM_26, SoccerRuleBooks.SSIM):
        return SSim2026Rules()

    # cases: FIFA and UNKNOWN
    return FIFASoccerRules()


@dataclass(frozen=True)
class SoccerRules:
    """Rule book for soccer games.

    Note: Use default values from official FIFA rule book if there exists an equivalent rule. In all other cases try to choose sensible values which work in conjunction with the official FIFA rule book.
    """

    max_team_size: int = 11  # officially 11 vs. 11 players
    """The maximum number of players per team."""

    max_player_no: int = 99  # not sure if this is an official restriction... but limiting it to two digits max is likely to simplify rendering, etc.
    """The maximum player number a player can choose."""

    half_time: int = 45 * 60  # officially 45min per half
    """The duration (in seconds) of a game half."""

    extra_half_time: int = 15 * 60  # officially 15min per extra half
    """The duration (in seconds) of an extra half."""

    kick_off_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the kick-off team has exclusive access to the ball."""

    throw_in_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the throw-in team has exclusive access to the ball."""

    corner_kick_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the corner-kick team has exclusive access to the ball."""

    goal_kick_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the goal-kick team has exclusive access to the ball."""

    free_kick_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the free-kick team has exclusive access to the ball."""

    direct_free_kick_time: int = 15  # unofficial - use negative values for disabling
    """The time (in seconds) the direct-free-kick team has exclusive access to the ball."""

    goal_pause_time: int = 3  # unofficial
    """The time (in seconds) to "pause" the game after a goal before switching to kick-off play mode for the opposite team."""

    throwin_wait_time: int = 1  # unofficial
    """The time (in seconds) to referee will wait after the ball left the field before switching to throw-in play mode."""

    default_field_version: SoccerFieldVersions = SoccerFieldVersions.FIFA  # official
    """The default field version to use with the rule book."""


@dataclass(frozen=True)
class FIFASoccerRules(SoccerRules):
    """Official FIFA soccer game rule book."""


@dataclass(frozen=True)
class SSim2026Rules(SoccerRules):
    """2026 version of the official RoboCup Soccer Simulation League rule book."""

    def __init__(self) -> None:
        """Construct a new RCSSim 2026 rule book."""

        super().__init__(
            max_team_size=7,
            half_time=5 * 60,
            extra_half_time=3 * 60,
            default_field_version=SoccerFieldVersions.FIFA_7_VS_7,
        )
