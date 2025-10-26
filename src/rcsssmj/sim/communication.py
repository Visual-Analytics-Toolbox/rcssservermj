import base64
import binascii
from collections import Counter
from math import pow, tanh
import random
from typing import Dict, List, Optional

from rcsssmj.sim.agent_id import AgentID
from rcsssmj.sim.sim_agent import SimAgent
from rcsssmj.sim.sim_atom import SimSite
from rcsssmj.sim.perceptions import HearPerception, Perception
from rcsssmj.utils.geometry import distance


class CommunicationController:
    """Communication controller in a simulation."""

    def __init__(self, max_message_length: int=20):
        """Construct a new communication controller.

        Parameter
        ---------
        max_message_length: int, default=20
            The maximum length of a message in bytes. Messages longer than this will be rejected.
        """

        self._max_message_length = max_message_length
        """The maximum length of a message in bytes. Messages longer than this will be rejected."""

        self._com_interfaces: Dict[AgentID, CommunicationInterface] = {}
        """A cached mapping of agent ids to their communication interfaces."""

    def update_com_interfaces(self, agents: List[SimAgent]) -> None:
        """Updates the internal cached dictionary of communication interfaces."""
        self._com_interfaces.clear()
        for agent in agents:
            for atom in agent.sim_atoms.values():
                if isinstance(atom, CommunicationInterface):
                    self._com_interfaces[agent.agent_id] = atom
                    break

    def _is_valid_message(self, message: str) -> bool:
        """Checks if an incoming message is valid."""
        try:
            decoded = base64.b64decode(message, validate=True)
            return len(decoded) <= 20
        except binascii.Error:
            return False

    def update(self, timestep: float, agents: List[SimAgent]) -> None:
        """Distributes the messages requested to send by the communication interfaces."""
        num_msgs = {}
        for agent_id, com_interface in self._com_interfaces.items():
            valid = com_interface.requested_broadcast is not None
            if valid:
                if not self._is_valid_message(com_interface.requested_broadcast):
                    valid = False
                    com_interface.requested_broadcast = None
            x = 1 if valid else 0
            if agent_id.team_id not in num_msgs:
                num_msgs[agent_id.team_id] = x
            else:
                num_msgs[agent_id.team_id] += x

            com_interface.clear_received_messages()

        for sender_agent_id, sender_com_interface in self._com_interfaces.items():
            if sender_com_interface.requested_broadcast is None:
                continue

            for target_agent_id, target_com_interface in self._com_interfaces.items():
                if target_agent_id == sender_agent_id:
                    continue

                # We assume that any other transmission is always strong enough to cause packet loss, no matter the distance
                tx_collision_loss_param = 0.02 / timestep
                packet_loss_rate_tx_collision = 1 - 1 / pow(2, tx_collision_loss_param * num_msgs[sender_agent_id.team_id])
                if random.random() <= packet_loss_rate_tx_collision:
                    continue

                player_distance = distance(sender_com_interface.xpos, target_com_interface.xpos)
                packet_loss_rate_distance = 0.5 * tanh(0.14 * player_distance - 2.7) + 0.5
                if random.random() <= packet_loss_rate_distance:
                    continue

                target_com_interface.deliver_message(sender_com_interface.requested_broadcast)
            sender_com_interface.requested_broadcast = None


class CommunicationInterface(SimSite):
    """Communication interface in a simulation."""

    def __init__(self, name: str):
        """Construct a new communication interface."""
        super().__init__(name)

        self.requested_broadcast: Optional[str] = None
        """The requested broadcast message (if any)."""

        self._received_messages: List[str] = []
        """The received messages from other agents."""

    def clear_received_messages(self) -> None:
        """Clears the buffer of received messages.
        No more old messages will be contained in the perception."""
        self._received_messages.clear()

    def deliver_message(self, message: str) -> None:
        """Delivers a message from another agent."""
        self._received_messages.append(message)
        
    def generate_perceptions(self, agent_perceptions: List[Perception]) -> None:
        for message in self._received_messages:
            agent_perceptions.append(HearPerception(message))
