import logging
from itertools import chain
from math import pi, sqrt
from typing import cast

import mujoco
import numpy as np

from rcsssmj.games.soccer.play_mode import PlayMode
from rcsssmj.games.soccer.sim.soccer_agent import SoccerAgent
from rcsssmj.games.soccer.sim.soccer_game import PSoccerGame
from rcsssmj.games.teams import TeamSide

logger = logging.getLogger(__name__)


class SoccerReferee:
    """A referee, applying soccer game rules."""

    def __init__(self, game: PSoccerGame | None = None) -> None:
        """Create a new soccer referee.

        Parameter
        ---------
        game: PSoccerGame | None, default=None
            The soccer game instance to referee. If ``None``, the game instance must be injected from externally before calling any method of the referee.
        """

        # MYPY-HACK: The game instance may initially be `None` but will be automatically injected in this case by the simulation (cast is used to silence mypy while preventing repetitive `None` checks)
        self.game: PSoccerGame = cast(PSoccerGame, game)
        """The soccer game instance to referee."""

        self._did_act: bool = False
        """Flag if the referee has already taken a decision in this referee cycle."""

    def reset(self) -> None:
        """Reinitialize the referee."""

        # init referee state
        self._did_act = False

    def is_beaming_allowed(self) -> bool:
        """Check if an agent is allowed to beam in the current game state."""

        return self.game.game_state.play_mode in (PlayMode.BEFORE_KICK_OFF, PlayMode.GOAL_LEFT, PlayMode.GOAL_RIGHT)

    def kick_off(self, team_side: TeamSide | int) -> None:
        """Instruct kickoff for the given team.

        Parameter
        ---------
        team_side: TeamSide | int
            The team side for which to give the kick off.
        """

        if not isinstance(team_side, TeamSide):
            team_side = TeamSide.from_id(team_side)

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = team_side

        self.game.ball.drop_at(0, 0)

    def play_on(self) -> None:
        """Instruct the normal progressing of the game."""

        self._did_act = True
        self.game.game_state.set_play_mode(PlayMode.PLAY_ON)

    def throw_in(self, team_side: TeamSide) -> None:
        """Instruct a throw in for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the throw in.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.THROW_IN_LEFT, PlayMode.THROW_IN_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        y = self.game.field.field_area.min_y if self.game.ball.xpos[1] < 0 else self.game.field.field_area.max_y
        self.game.ball.drop_at(self.game.ball.xpos[0], y)

    def corner_kick(self, team_side: TeamSide) -> None:
        """Instruct corner kick for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the corner kick.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.CORNER_KICK_LEFT, PlayMode.CORNER_KICK_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        x = self.game.field.field_area.max_x if team_side == TeamSide.LEFT else self.game.field.field_area.min_x
        y = self.game.field.field_area.min_y if self.game.ball.xpos[1] < 0 else self.game.field.field_area.max_y
        self.game.ball.drop_at(x, y)

    def goal_kick(self, team_side: TeamSide) -> None:
        """Instruct goal kick for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the goal kick.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.GOAL_KICK_LEFT, PlayMode.GOAL_KICK_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        drop_pos = self.game.field.left_goalie_area.center() if team_side == TeamSide.LEFT else self.game.field.right_goalie_area.center()
        self.game.ball.drop_at(drop_pos[0], drop_pos[1])

    def offsite(self, team_side: TeamSide) -> None:
        """Offsite state for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the offside.
        """

        self._did_act = True
        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.OFFSIDE_LEFT, PlayMode.OFFSIDE_RIGHT)

    def game_over(self) -> None:
        """Instruct the end of the game."""

        self._did_act = True

        self.game.game_state.set_play_mode(PlayMode.GAME_OVER)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        self.game.ball.reset_contacts()

    def goal(self, team_side: TeamSide) -> None:
        """Count a goal for the given team and set the play mode accordingly.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the goal.
        """

        self._did_act = True

        self.game.game_state.goal(team_side)

    def free_kick(self, team_side: TeamSide) -> None:
        """Instruct an indirect free kick for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the free kick.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.FREE_KICK_LEFT, PlayMode.FREE_KICK_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = team_side

        self.game.ball.drop()

    def direct_free_kick(self, team_side: TeamSide) -> None:
        """Instruct a direct free kick for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the direct free kick.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.DIRECT_FREE_KICK_LEFT, PlayMode.DIRECT_FREE_KICK_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        self.game.ball.drop()

    def penalty_kick(self, team_side: TeamSide) -> None:
        """Instruct a penalty kick for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the penalty kick.
        """

        self._did_act = True

        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.PENALTY_KICK_LEFT, PlayMode.PENALTY_KICK_RIGHT)
        self.game.game_state.agent_na_touch_ball = None
        self.game.game_state.team_na_score = None

        penalty_spot_x = self.game.field.field_area.max_x - self.game.field.penalty_spot_distance
        self.game.ball.drop_at(-penalty_spot_x if team_side == TeamSide.LEFT else penalty_spot_x)

    def penalty_shoot(self, team_side: TeamSide) -> None:
        """Instruct a penalty shoot for the given team.

        Parameter
        ---------
        team_side: TeamSide
            The team side for which to give the penalty shoot.
        """

        self._did_act = True
        self.game.game_state.set_play_mode_for_team(team_side, PlayMode.PENALTY_SHOOT_LEFT, PlayMode.PENALTY_SHOOT_RIGHT)
        # TODO: place ball for penalty-shoot

    def drop_ball(self, pos: tuple[float, float] | None = None) -> None:
        """Drop the ball at the specified position and instruct the normal progressing of the game.

        Parameter
        ---------
        pos: tuple[float, float] | None, default=None
            The position at which to drop the ball or none, to drop it at its current location.
        """

        self._did_act = True
        if pos is not None:
            self.game.ball.drop_at(pos[0], pos[1])
        else:
            self.game.ball.drop()
        # TODO: cause relocation of all agents nearby the ball (within a radius defined here)

        self.game.game_state.set_play_mode(PlayMode.PLAY_ON)

    def referee(self) -> None:
        """Referee the current game situation."""

        # update game state times
        self.game.game_state.update(self.game.sim_time, progress_play_time=self.game.game_state.play_mode not in (PlayMode.BEFORE_KICK_OFF, PlayMode.GAME_OVER))

        # check game over
        if self.game.game_state.play_time >= self.game.rules.get_end_time_for(self.game.game_state.phase):
            self.game_over()
            return

        # check for rule violations
        self._check_fouls()

        # automatically progress play mode based on timeouts, object locations and action triggers
        self._check_timeouts()
        self._check_location_triggers()
        self._check_contact_triggers()

        # penalize misplaced players (according to play mode restrictions)
        self._penalize_misplaced_players()

        # reset decision flag
        self._did_act = False

    def _check_fouls(self) -> None:
        """Check fouls / violations of game rules."""

        active_ball_contact = self.game.ball.active_contact
        last_ball_contact = self.game.ball.last_contact
        last_ball_contact_change = self.game.ball.contact_change
        agent_na_touch_ball = self.game.game_state.agent_na_touch_ball
        sim_time = self.game.game_state.sim_time

        # check no score rule
        if self.game.game_state.team_na_score is not None and active_ball_contact is not None and last_ball_contact is not None:
            self.game.game_state.team_na_score = None
        
        # check hand touch foul
        if self.__check_hand_foul():
            self.direct_free_kick(TeamSide.get_opposing_side(active_ball_contact.team_id))
        
        # check double-touch rule
        if agent_na_touch_ball is not None and active_ball_contact is not None and last_ball_contact is not None:
            if agent_na_touch_ball == last_ball_contact and agent_na_touch_ball == active_ball_contact:
                if last_ball_contact_change is not None and sim_time - last_ball_contact_change > 1:
                    self.free_kick(TeamSide.get_opposing_side(agent_na_touch_ball.team_id))
                    return
            else:
                self.game.game_state.agent_na_touch_ball = None

    def __check_hand_foul(self) -> bool:
        active_ball_contact = self.game.ball.active_contact
        body_parts = self.game.ball.active_contact_body_parts

        # Check if the ball is touching any hand or forearm
        if body_parts is not None and any('hand' in part or 'forearm' in part for part in body_parts):
            logger.info('='*20)
            logger.info('Ball is in contact with hand or forearm')

            # Exceptions
            agent = self.game.get_players(active_ball_contact.team_id)[active_ball_contact.player_no]
            mj_model = getattr(self.game, 'mj_model', None)
            mj_data = getattr(self.game, 'mj_data', None)

            is_foul = True

            if mj_model is not None and mj_data is not None:
                # Exception: Goalie touching the ball inside their own penalty area
                if agent.agent_id.player_no == 1:
                    ball_x, ball_y = self.game.ball.xpos[0], self.game.ball.xpos[1]
                    if agent.agent_id.team_id == TeamSide.LEFT.value:
                        if self.game.field.left_penalty_area.contains(ball_x, ball_y):
                            is_foul = False
                    elif agent.agent_id.team_id == TeamSide.RIGHT.value:
                        if self.game.field.right_penalty_area.contains(ball_x, ball_y): 
                            is_foul = False

                if is_foul:
                    # Retrieve the touched hand/forearm geometries and their dimensions
                    hand_geoms = []
                    for part in body_parts:
                        gid = agent.hand_geom_ids.get(part, -1)
                        if gid >= 0:
                            hand_pos = np.array(mj_data.geom_xpos[gid])
                            h_len = max(mj_model.geom_size[gid][0], mj_model.geom_size[gid][1])
                            h_rad = min(mj_model.geom_size[gid][0], mj_model.geom_size[gid][1])
                            hand_geoms.append((hand_pos, h_len, h_rad, part))
                    # If valid geometries were found, check for morphological extensions
                    if hand_geoms:

                        # Retrieve the torso's bounding box and position as the reference center
                        torso_geom_id = agent.torso_geom_id
                        if torso_geom_id >= 0:
                            torso_x_size = mj_model.geom_size[torso_geom_id][0] 
                            torso_y_size = mj_model.geom_size[torso_geom_id][1] 
                            torso_pos = np.array(mj_data.geom_xpos[torso_geom_id])

                        # Determine the top of the robot's head to calculate vertical extensions
                        head_pos = torso_pos + np.array([0.0, 0.0, 0.25]) # Fallback
                        head_top_z = head_pos[2] + 0.10 # Fallback

                        camera_site_id = agent.camera_site_id
                        if camera_site_id >= 0:
                            head_pos = np.array(mj_data.site_xpos[camera_site_id])
                            head_top_z = head_pos[2]
                            
                            # Find the body that owns the camera (the Head)
                            head_body_id = mj_model.site_bodyid[camera_site_id]
                            geom_adr = mj_model.body_geomadr[head_body_id]
                            geom_num = mj_model.body_geomnum[head_body_id]
                            
                            # Scan all head geometries to find the absolute highest point (Z-axis)
                            for i in range(geom_num):
                                gid = geom_adr + i
                                gpos = mj_data.geom_xpos[gid]
                                gsize = mj_model.geom_size[gid]
                                top_z = gpos[2] + max(gsize)
                                if top_z > head_top_z:
                                    head_top_z = top_z

                        head_z = head_pos[2]
                        is_fallen = head_z < 0.45

                        is_extended = False
                        extended_x = False
                        extended_y = False
                        extended_z = False

                        torso_body_id = agent.torso_body_id
                        torso_xmat = np.array(mj_data.xmat[torso_body_id]).reshape(3, 3)

                        if is_fallen:
                            head_local = torso_xmat.T @ (head_pos - torso_pos)

                            for hp, h_len, h_rad, part in hand_geoms:
                                hand_local = torso_xmat.T @ (hp - torso_pos)
                                
                                # 1. Hand raised high in the air (Global Z)
                                tolerancia_z_global = torso_pos[2] + (max(torso_x_size, torso_y_size) * 2)
                                ext_z_global = hp[2] > tolerancia_z_global
                                
                                # 2. Hand stretched above the head (Local Z)
                                # Only penalized if fallen on the side (shoulder-to-shoulder axis points up)
                                is_fallen_on_side = abs(torso_xmat[2, 1]) > abs(torso_xmat[2, 0])
                                ext_torso_head = is_fallen_on_side and (hand_local[2] > abs(head_local[2]))
                                
                                # Note: Arms extended to the sides or front (Local X and Y) are ignored 
                                # since they are legitimately used for ground support when fallen.
                                
                                if ext_z_global: extended_y = True   # Mapped to Y log for ground Z extension
                                if ext_torso_head: extended_z = True # Local Z (above the head)
                                
                                if ext_z_global or ext_torso_head:
                                    is_extended = True
                                    break

                        else:

                            for hp, h_len, h_rad, part in hand_geoms:
                                hand_local = torso_xmat.T @ (hp - torso_pos)
                                
                                # Dynamic extension tolerance based on robot size and arm radius
                                MAX_FRONTAL_ARM_WIDTHS = 3
                                MAX_LATERAL_ARM_WIDTHS = 5

                                tolerancia_x_dinamica = torso_x_size + (h_rad * MAX_FRONTAL_ARM_WIDTHS)
                                tolerancia_y_dinamica = torso_y_size + (h_rad * MAX_LATERAL_ARM_WIDTHS)

                                # Check extensions in the torso's local X/Y axes or above the head (Global Z)
                                ext_x = abs(hand_local[0]) > tolerancia_x_dinamica
                                ext_y = abs(hand_local[1]) > tolerancia_y_dinamica
                                ext_z = hp[2] > head_top_z
                                
                                if ext_x: extended_x = True
                                if ext_y: extended_y = True
                                if ext_z: extended_z = True
                                
                                if ext_x or ext_y or ext_z:
                                    is_extended = True
                                    break

                        logger.info('X-Axis Extension (Front/Back): %s', extended_x)
                        logger.info('Y-Axis Extension (Lateral): %s', extended_y)
                        logger.info('Z-Axis Extension (Above Head): %s', extended_z)
                        logger.info('is_extended: %s', is_extended)
                                             
                        if not is_extended:
                            is_foul = False

            logger.info('is_foul: %s', is_foul)
            return is_foul
            
        return False

    def _check_timeouts(self) -> None:
        """Check timeouts (kick-off time, throw-in time, etc.) for the current play mode."""

        if self._did_act:
            # the referee has already taken a decision in this simulation cycle
            return

        pm = self.game.game_state.play_mode

        if pm == PlayMode.PLAY_ON:
            # shortcut, as remaining rules only apply to other states than play-on
            return

        def check_timeout(timeout: int, *play_modes: PlayMode) -> bool:
            """Helper function for checking a play mode specific timeout."""
            return timeout >= 0 and pm in play_modes and self.game.game_state.get_play_mode_age() > timeout

        # check kick-off, throw-in, corner-kick and free-kick times
        if (
            check_timeout(self.game.rules.kick_off_time, PlayMode.KICK_OFF_LEFT, PlayMode.KICK_OFF_RIGHT)
            or check_timeout(self.game.rules.throw_in_time, PlayMode.THROW_IN_LEFT, PlayMode.THROW_IN_RIGHT)
            or check_timeout(self.game.rules.corner_kick_time, PlayMode.CORNER_KICK_LEFT, PlayMode.CORNER_KICK_RIGHT)
            or check_timeout(self.game.rules.free_kick_time, PlayMode.FREE_KICK_LEFT, PlayMode.FREE_KICK_RIGHT)
            or check_timeout(self.game.rules.direct_free_kick_time, PlayMode.DIRECT_FREE_KICK_LEFT, PlayMode.DIRECT_FREE_KICK_RIGHT)
        ):
            self.play_on()
            return

        # check goal-kick times
        if check_timeout(self.game.rules.goal_kick_time, PlayMode.GOAL_KICK_LEFT):
            # drop ball at a corner of the left goalie area
            self.drop_ball((self.game.field.left_goalie_area.max_x, self.game.field.left_goalie_area.max_y))
            return

        if check_timeout(self.game.rules.goal_kick_time, PlayMode.GOAL_KICK_RIGHT):
            self.drop_ball((self.game.field.right_goalie_area.min_x, self.game.field.right_goalie_area.max_y))
            return

        # check goal pause time
        if pm == PlayMode.GOAL_LEFT and self.game.game_state.get_play_mode_age() > self.game.rules.goal_pause_time:
            self.kick_off(TeamSide.RIGHT)
            return

        if pm == PlayMode.GOAL_RIGHT and self.game.game_state.get_play_mode_age() > self.game.rules.goal_pause_time:
            self.kick_off(TeamSide.LEFT)
            return

    def _check_location_triggers(self) -> None:
        """Check location triggers (ball leaving the field in play-on, leaving the goalie-area in goal-kick, etc.) for the current play mode."""

        if self._did_act:
            # the referee has already taken a decision in this simulation cycle
            return

        pm = self.game.game_state.play_mode

        if pm in (PlayMode.GOAL_LEFT, PlayMode.GOAL_RIGHT):
            # no location triggers in goal states
            return

        # check left goal
        if self.game.field.left_goal_box.contains(self.game.ball.xpos[0], self.game.ball.xpos[1], self.game.ball.xpos[2]):
            if pm == PlayMode.GOAL_KICK_LEFT:
                # drop ball at a corner of the left goalie area
                self.drop_ball((self.game.field.left_goalie_area.max_x, self.game.field.left_goalie_area.max_y))
            elif self.game.game_state.team_na_score == TeamSide.RIGHT:
                self.goal_kick(TeamSide.LEFT)
            else:
                self.goal(TeamSide.RIGHT)
            return

        # check right goal
        if self.game.field.right_goal_box.contains(self.game.ball.xpos[0], self.game.ball.xpos[1], self.game.ball.xpos[2]):
            if pm == PlayMode.GOAL_KICK_RIGHT:
                # drop ball at a corner of the left goalie area
                self.drop_ball((self.game.field.right_goalie_area.min_x, self.game.field.right_goalie_area.max_y))
            elif self.game.game_state.team_na_score == TeamSide.LEFT:
                self.goal_kick(TeamSide.RIGHT)
            else:
                self.goal(TeamSide.LEFT)
            return

        # check if the ball left the field
        if not self.game.field.field_area.contains(self.game.ball.xpos[0], self.game.ball.xpos[1]):
            agent_contact = self.game.ball.get_most_recent_contact()
            last_team_contact = TeamSide.UNKNOWN if agent_contact is None else TeamSide.from_id(agent_contact.team_id)

            if self.game.ball.xpos[0] < self.game.field.field_area.min_x:
                # corner-kick right / goal-kick left
                if last_team_contact == TeamSide.LEFT:
                    self.corner_kick(TeamSide.RIGHT)
                else:
                    self.goal_kick(TeamSide.LEFT)

            elif self.game.ball.xpos[0] > self.game.field.field_area.max_x:
                # corner-kick left / goal-kick right
                if last_team_contact == TeamSide.RIGHT:
                    self.corner_kick(TeamSide.LEFT)
                else:
                    self.goal_kick(TeamSide.RIGHT)

            # elif self.game.ball.xpos[1] < -self.game.field.field_area.min_y or self.game.ball.xpos[1] > self.game.field.field_area.max_y:
            else:
                # throw-in
                self.throw_in(TeamSide.get_opposing_side(last_team_contact))

            return

        # check if ball left the goalie area on goal-kick
        if (pm == PlayMode.GOAL_KICK_LEFT and not self.game.field.left_goalie_area.contains(self.game.ball.xpos[0], self.game.ball.xpos[1])) or (
            pm == PlayMode.GOAL_KICK_RIGHT and not self.game.field.right_goalie_area.contains(self.game.ball.xpos[0], self.game.ball.xpos[1])
        ):
            self.play_on()
            return

    def _check_contact_triggers(self) -> None:
        """Check contact action triggers (touching the ball in kick-off, throw-in, etc.) for the current play mode."""

        if self._did_act:
            # the referee has already taken a decision in this simulation cycle
            return

        if self.game.ball.active_contact is None:
            # no action trigger
            return

        pm = self.game.game_state.play_mode

        if pm == PlayMode.PLAY_ON:
            # shortcut, as remaining rules only apply to other states than play-on
            return

        # kick-off, throw-in, corner-kick and free kicks
        if pm in (
            PlayMode.KICK_OFF_LEFT,
            PlayMode.KICK_OFF_RIGHT,
            PlayMode.THROW_IN_LEFT,
            PlayMode.THROW_IN_RIGHT,
            PlayMode.CORNER_KICK_LEFT,
            PlayMode.CORNER_KICK_RIGHT,
            PlayMode.FREE_KICK_LEFT,
            PlayMode.FREE_KICK_RIGHT,
            PlayMode.DIRECT_FREE_KICK_LEFT,
            PlayMode.DIRECT_FREE_KICK_RIGHT,
        ):
            if len(self.game.get_players(self.game.ball.active_contact.team_id)) > 1:
                self.game.game_state.agent_na_touch_ball = self.game.ball.active_contact

            self.play_on()
            return

    def _penalize_misplaced_players(self) -> None:
        """Check if players are within areas they are not allowed in and, in case, penalize them accordingly."""

        pm = self.game.game_state.play_mode

        if pm == PlayMode.KICK_OFF_LEFT:
            self._check_placement_for_kick_off_left()

        elif pm == PlayMode.KICK_OFF_RIGHT:
            self._check_placement_for_kick_off_right()

        elif pm == PlayMode.GOAL_KICK_LEFT:
            self._check_placement_for_goal_kick_left()

        elif pm == PlayMode.GOAL_KICK_RIGHT:
            self._check_placement_for_goal_kick_right()

        elif pm in (PlayMode.THROW_IN_LEFT, PlayMode.CORNER_KICK_LEFT, PlayMode.FREE_KICK_LEFT, PlayMode.DIRECT_FREE_KICK_LEFT):
            self._check_placement_for_free_kick_left()

        elif pm in (PlayMode.THROW_IN_RIGHT, PlayMode.CORNER_KICK_RIGHT, PlayMode.FREE_KICK_RIGHT, PlayMode.DIRECT_FREE_KICK_RIGHT):
            self._check_placement_for_free_kick_right()

    def _check_placement_for_kick_off_left(self) -> None:
        """Penalize all players of the left team that are on the right side and all players of the right team that are on the left side or within the middle circle."""

        cc_radius = self.game.field.center_circle_radius
        kickoff_agent = None
        kickoff_agent_dist = cc_radius
        for agent in self.game.left_players.values():
            if agent.xpos[0] > 0:
                # agent is in opponent half --> check if it is the closest agent to the center
                dist_to_center = sqrt(agent.xpos[0] ** 2 + agent.xpos[1] ** 2)
                if dist_to_center > kickoff_agent_dist:
                    # not kickoff agent or outside the center circle --> penalize
                    self._penalize(agent)
                else:
                    # closer to center as the kickoff agent --> penalize previously considered kickoff agent
                    if kickoff_agent is not None:
                        self._penalize(kickoff_agent)

                    kickoff_agent = agent
                    kickoff_agent_dist = dist_to_center

        for agent in self.game.right_players.values():
            if agent.xpos[0] < 0 or sqrt(agent.xpos[0] ** 2 + agent.xpos[1] ** 2) < cc_radius:
                self._penalize(agent)

    def _check_placement_for_kick_off_right(self) -> None:
        """Penalize all players of the left team that are on the right side or within the middle circle and all players of the right team that are on the left side."""

        cc_radius = self.game.field.center_circle_radius
        for agent in self.game.left_players.values():
            if agent.xpos[0] > 0 or sqrt(agent.xpos[0] ** 2 + agent.xpos[1] ** 2) < cc_radius:
                self._penalize(agent)

        kickoff_agent = None
        kickoff_agent_dist = cc_radius
        for agent in self.game.right_players.values():
            if agent.xpos[0] < 0:
                # agent is in opponent half --> check if it is the closest agent to the center
                dist_to_center = sqrt(agent.xpos[0] ** 2 + agent.xpos[1] ** 2)
                if dist_to_center > kickoff_agent_dist:
                    # not kickoff agent or outside the center circle --> penalize
                    self._penalize(agent)
                else:
                    # closer to center as the kickoff agent --> penalize previously considered kickoff agent
                    if kickoff_agent is not None:
                        self._penalize(kickoff_agent)

                    kickoff_agent = agent
                    kickoff_agent_dist = dist_to_center

    def _check_placement_for_goal_kick_left(self) -> None:
        """Penalize all players of the right team that are within the left goalie area."""

        area = self.game.field.left_goalie_area

        for agent in self.game.right_players.values():
            if area.contains(agent.xpos[0], agent.xpos[1]):
                self._penalize(agent)

    def _check_placement_for_goal_kick_right(self) -> None:
        """Penalize all players of the left team that are within the right goalie area."""

        area = self.game.field.right_goalie_area

        for agent in self.game.left_players.values():
            if area.contains(agent.xpos[0], agent.xpos[1]):
                self._penalize(agent)

    def _check_placement_for_free_kick_left(self) -> None:
        """Penalize all players of the right team that are within the center circle radius to the ball."""

        ball_x = self.game.ball.xpos[0]
        ball_y = self.game.ball.xpos[1]
        cc_radius = self.game.field.center_circle_radius

        for agent in self.game.right_players.values():
            if sqrt((agent.xpos[0] - ball_x) ** 2 + (agent.xpos[1] - ball_y) ** 2) < cc_radius:
                self._penalize(agent)

    def _check_placement_for_free_kick_right(self) -> None:
        """Penalize all players of the left team that are within the center circle radius to the ball."""

        ball_x = self.game.ball.xpos[0]
        ball_y = self.game.ball.xpos[1]
        cc_radius = self.game.field.center_circle_radius

        for agent in self.game.left_players.values():
            if sqrt((agent.xpos[0] - ball_x) ** 2 + (agent.xpos[1] - ball_y) ** 2) < cc_radius:
                self._penalize(agent)

    def _penalize(self, player: SoccerAgent) -> None:
        """Penalize the given player."""

        # choose upper / lower field border based on ball position
        y_side = -1 if self.game.ball.xpos[1] > 0 else 1

        x = 0.0
        y = y_side * ((self.game.field.field_dim[1] / 2) + self.game.field.field_border)

        # check for free space in 2m steps
        x_step = -2.0 if player.agent_id.team_id == TeamSide.LEFT.value else 2.0
        is_occupied = True
        while is_occupied:
            x += x_step

            # check location for collisions
            # TODO: Extract other player locations once and vectorize distance calculations (should be more efficient even if the python for-loop may be early aborted).
            for agent in chain(self.game.left_players.values(), self.game.right_players.values()):
                if agent != player and sqrt((agent.xpos[0] - x) ** 2 + (agent.xpos[1] - y) ** 2) < 1.0:
                    # other player is occupying the location already
                    break
            else:
                is_occupied = False

        player.drop_at(x, y, -pi * y_side / 2)


class KickChallengeReferee(SoccerReferee):
    """A referee, applying soccer game rules for a kick-challenge."""

    def __init__(self, game: PSoccerGame | None = None) -> None:
        """Create a new kick challenge referee.

        Parameter
        ---------
        game: PSoccerGame | None, default=None
            The soccer game instance to referee. If ``None``, the game instance must be injected from externally before calling any method of the referee.
        """

        super().__init__(game)

        self._start_time: float = -1
        """The time at which to automatically start the game."""

    def reset(self) -> None:
        super().reset()

        self._start_time = -1

    def referee(self) -> None:
        # check if an agent is connected
        if self._start_time < 0 and len(self.game.get_players(TeamSide.LEFT)) > 0:
            self._start_time = self.game.game_state.sim_time + 2

        # automatically kick-off left 2 seconds after the first agent connected
        if self.game.game_state.play_mode == PlayMode.BEFORE_KICK_OFF and self._start_time >= 0 and self._start_time <= self.game.game_state.sim_time:
            self.kick_off(TeamSide.LEFT)

        # forward to base class
        super().referee()

        # calculate challenge score
        if self.game.game_state.play_mode != PlayMode.GAME_OVER:
            score = int((self.game.ball.xpos[0] - abs(self.game.ball.xpos[1])) * 100)
            self.game.game_state.set_score(TeamSide.LEFT, score)

    def _check_fouls(self) -> None:
        # disable no score rule
        self.game.game_state.team_na_score = None

        # check double-touch rule
        if self.game.ball.active_contact is not None and self.game.ball.active_contact == self.game.ball.last_contact:
            self.game_over()

    def _check_location_triggers(self) -> None:
        # no location triggers in challenge
        return

    def _penalize_misplaced_players(self) -> None:
        # no misplacement of agents in challenges
        return
