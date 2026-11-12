"""Shared lifecycle, public-observation helpers, and validated policy selection."""

from typing import get_args

from simulator import CompanyId, Observation, PlayerId

from ..base import ActionMask, Player, PlayerKind


class RuleBasedPlayer(Player):
    """Share the decision interface across handwritten policies and baselines.

    Strategies use conditionals or action scores over the observation and mask.
    Shared helpers count own cards and predict token acquisition from public
    information. Strategies may retain player-visible context. RandomPlayer
    and RankingPlayer own private RNGs and override reset to manage them.
    The lifecycle, helpers, and decision wrapper are implemented;
    policy must be implemented or overridden before choosing moves.
    """

    def __init__(self, player_id: PlayerId, name: str = "Rule-based player") -> None:
        """Validate and store identity through Player without creating game state."""
        super().__init__(player_id, name)

    @property
    def kind(self) -> PlayerKind:
        """Identify this player as a handwritten strategy or rule-based baseline."""
        return "rule_based"

    def reset(self, seed: int | None = None) -> None:
        """Preserve identity and ignore seed through the base reset.

        There is currently no strategy context to clear. A future strategy that
        retains observations or derived information must extend this method to
        clear that context between games. Do not reset or advance a simulator.
        """
        super().reset(seed)

    async def choose_action(self, observation: Observation, mask: ActionMask) -> int:
        """Validate the request, call policy, and validate its selected action ID.

        Policy evaluation is synchronous; the async wrapper gives
        the game loop the same interface used for neural and human players.
        Pass the supplied observation and mask to policy once. Normalize its
        result to a Python int and reject invalid or masked IDs with ValueError;
        never replace a bad choice. Policy exceptions, including the placeholder's
        NotImplementedError, propagate to the caller. Do not call on_observation
        or advance the game here.
        """
        self._validate_request(observation, mask)
        action_id = self.policy(observation, mask)
        return self._validate_action_id(action_id, mask)

    def count_hand_cards(self, observation: Observation) -> dict[CompanyId, int]:
        """Return fresh hand counts for companies 5 through 10, including zeros.

        Validate the observation's viewer using _validate_observation, without
        requiring an active turn. Count only the viewer's exact hand, preserving
        duplicate copies. Exclude investments and public known-hand counts.
        In play phase the hand includes the newly taken card. Do not retain or
        mutate observations.
        """
        self._validate_observation(observation)
        counts: dict[CompanyId, int] = dict.fromkeys(get_args(CompanyId), 0)
        for company_id in observation.hand:
            counts[company_id] += 1
        return counts

    def count_investment_cards(self, observation: Observation) -> dict[CompanyId, int]:
        """Return fresh own-investment counts for companies 5 to 10, including zeros.

        Validate the observation's viewer without requiring an active turn.
        Find this player's public state by player_id and count each invested
        card, including duplicate copies. Exclude the hand, opponents, market,
        and known-hand counts. Raise ValueError if this player's public state
        is missing. Do not retain or mutate observations.
        """
        self._validate_observation(observation)
        own_state = next(
            (state for state in observation.players if state.player_id == self.player_id),
            None,
        )
        if own_state is None:
            raise ValueError("observation must include this player's public state")
        counts: dict[CompanyId, int] = dict.fromkeys(get_args(CompanyId), 0)
        for company_id in own_state.investments:
            counts[company_id] += 1
        return counts

    def count_own_cards(self, observation: Observation) -> dict[CompanyId, int]:
        """Return this player's hand-plus-investment counts for companies 5 to 10.

        Add count_hand_cards and count_investment_cards for this player's view.
        Include zero counts and propagate their observation validation errors.
        Count individual cards, not distinct company types. Exclude opponents'
        holdings, market cards, and known-hand counts; never count the same hand
        cards twice. Return a fresh dictionary without changing any state.
        """
        counts = self.count_hand_cards(observation)
        for company_id, investment_count in self.count_investment_cards(observation).items():
            counts[company_id] += investment_count
        return counts

    def would_acquire_anti_monopoly(self, observation: Observation, company_id: CompanyId) -> bool:
        """Predict whether investing one card would give this player a new token.

        Use only public investments and current token ownership. A first
        investment claims an unowned token. Taking a token from another player
        requires strictly exceeding that owner's invested count; a tie leaves
        it with them. Return False when this player already owns the token.

        Validate the viewer and company ID. This is a hypothetical calculation,
        not a legality check: the caller decides whether investing is available.
        Exclude hands, public known-hand counts, and final hand revelation.
        Change no observation, token, or simulator state.
        """
        self._validate_observation(observation)
        if type(company_id) is not int or company_id not in get_args(CompanyId):
            raise ValueError("company_id must be an integer from 5 through 10")
        owner = dict(observation.anti_monopoly_owners)[company_id]
        if owner == self.player_id:
            return False
        if owner is None:
            return True
        owner_state = next((state for state in observation.players if state.player_id == owner), None)
        if owner_state is None:
            raise ValueError("observation must include the token owner's public state")
        own_count = self.count_investment_cards(observation)[company_id]
        return own_count + 1 > owner_state.investments.count(company_id)

    def policy(self, observation: Observation, mask: ActionMask) -> int:
        """Apply the strategy's decision rules and return an allowed action ID.

        Initially inspect the current observation. Future strategies may also
        use context retained through on_observation. Use ActionCodec to translate
        readable actions into IDs or decode legal IDs when examining market entries.
        Each strategy defines tie-breaking, including seeded random selection.
        The particular strategy will be implemented separately.
        """
        raise NotImplementedError
