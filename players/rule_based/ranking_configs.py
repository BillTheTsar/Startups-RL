"""Named weighted-ranking variants: add more dictionaries here, not player classes.

Each entry overrides fields of RankingWeights; omitted fields use defaults.
`RankingPlayer1` preserves the original weights. Numbered dictionary profiles
change only weights, with no new events or lookahead:

* RankingPlayer2: new-token penalty -2 -> -1; invested-dominance penalty -2 -> -1;
  pickup commitment per investment +1 -> +2.
* RankingPlayer3: new-token penalty -2 -> -3; offer-risk penalty per remaining card
  -1 -> -2; value per coin 1/2 -> 3/4.

RankingPlayer5..19 explore cash, hand groups, majority/minority pressure,
company preferences, investment commitment, token tolerance, and offer risk.
Each dictionary below documents its intended emphasis; these are interacting
preferences, not guaranteed behaviors or measured improvements. In particular,
hand/group bonuses also discourage playing cards that would lose those bonuses.
All overrides are relative to RankingWeights defaults, NOT the preceding entry.
Tables use company IDs 5..10, hand counts 0..4, and majority gaps <=-2..>=+2.
Use Fraction for noninteger values so score ties remain exact.

Profile numbers do not indicate measured playing strength. Configurations
are copied/validated when constructing players. Each instance owns its own RNG.
The opponent registry includes every entry when imported; restart the CLI after
adding or editing entries. Existing players keep their frozen weights.
"""

from fractions import Fraction

from simulator import PlayerId

from .weighted_ranking_player import WeightedRankingPlayer


RANKING_CONFIGS: dict[str, dict[str, object]] = {
    # Original baseline: use every default weight without overrides.
    "RankingPlayer1": {},
    # Bolder investor: soften token/dominance penalties and favor pickups in
    # companies already invested in.
    "RankingPlayer2": {
        "new_token_value": -1,
        "dominated_investment_value": -1,
        "commitment_per_investment": 2,
    },
    # Cautious cash seeker: value coins more and penalize new tokens and risky
    # offers more strongly.
    "RankingPlayer3": {
        "new_token_value": -3,
        "offer_risk_per_card": -2,
        "coin_value": Fraction(3, 4),
    },
    # Committed generalist: no inherent hand preference between companies;
    # strengthen pickup commitment and offer protection while tolerating tokens.
    "RankingPlayer4": {
        "new_token_value": -1,
        "commitment_per_investment": Fraction(3, 2),
        "offer_risk_per_card": Fraction(-3, 2),
        "hand_values": (0, 0, 0, 0, 0, 0),
        "offer_risk_per_card": -2,
    },
    # Cash opportunist: prioritize market coins and avoiding deck fees; accept
    # weaker hand groups and less commitment to already-invested companies.
    "RankingPlayer5": {
        "coin_value": 1,
        "group_values": (0, 0, 1, 2, 3),
        "commitment_per_investment": Fraction(1, 2),
    },
    # Set collector: pay more readily to build/preserve pairs and larger groups.
    "RankingPlayer6": {
        "coin_value": Fraction(1, 4),
        "group_values": (0, 0, 2, 5, 8),
        "commitment_per_investment": Fraction(3, 2),
    },
    # Majority pursuer: reward visible leads strongly and tolerate new tokens;
    # resist offers that would leave a vulnerable holding behind.
    "RankingPlayer7": {
        "majority_values": (-4, -2, 0, 2, 6),
        "new_token_value": -1,
        "offer_risk_per_card": -2,
    },
    # Minority escape: strongly dislike trailing holdings; less sunk commitment
    # makes abandoning a weak company more attractive than chasing it forever.
    "RankingPlayer8": {
        "majority_values": (-8, -4, 0, 1, 3),
        "commitment_per_investment": Fraction(1, 2),
        "new_token_value": -3,
    },
    # Token tolerant: no acquisition penalty and a smaller dominance penalty.
    # Still no positive token reward or forecast of future fee exemptions.
    "RankingPlayer9": {
        "new_token_value": 0,
        "dominated_investment_value": -1,
        "coin_value": Fraction(1, 4),
    },
    # Token shy: strongly discourage acquisition and further dominant investment,
    # while valuing cash. Reward a clear lead enough that protecting it can beat
    # avoidance. Unlike a threshold chicken, this is never a hard veto.
    "RankingPlayer10": {
        "new_token_value": -4,
        "dominated_investment_value": -3,
        "coin_value": Fraction(3, 4),
        "offer_risk_per_card": -2,
        "majority_values": (-4, -2, 0, 1, 5),
    },
    # Small-company collector: favor holding 5/6 over 9/10, supported by groups.
    # Hand-only values encourage collecting AND retaining these cards in hand.
    "RankingPlayer11": {
        "hand_values": (2, 2, 0, 0, -2, -2),
        "group_values": (0, 0, 2, 4, 6),
        "commitment_per_investment": Fraction(1, 2),
    },
    # Large-company collector: reverse the usual hand preference toward 9/10;
    # soften the token penalty, but keep normal majority/excess considerations.
    "RankingPlayer12": {
        "hand_values": (-1, -1, 0, 0, 1, 1),
        "group_values": (0, 0, 2, 4, 6),
        "new_token_value": -1,
    },
    # Committed collector: pursue existing investments even beyond the normal
    # combined-holdings limit; this deliberately risks overconcentration.
    "RankingPlayer13": {
        "commitment_per_investment": 3,
        "coin_value": Fraction(1, 4),
        "excess_card_value": 0,
        "dominated_investment_value": -1,
        "new_token_value": -1,
    },
    # Flexible collector: no pickup bonus for past investments; resist excess
    # holdings and prefer cash when collection opportunities are comparable.
    "RankingPlayer14": {
        "commitment_per_investment": 0,
        "excess_card_value": -2,
        "dominated_investment_value": -3,
        "coin_value": Fraction(3, 4),
    },
    # Pair builder: a large first-pair bonus with only small gains for third and
    # fourth copies; remove inherent company preference to emphasize that shape.
    "RankingPlayer15": {
        "group_values": (0, 0, 3, 4, 5),
        "hand_values": (0, 0, 0, 0, 0, 0),
        "coin_value": Fraction(3, 4),
    },
    # Triple builder: the main group reward arrives on the third copy; remove
    # excess penalties so small companies can still be collected aggressively.
    "RankingPlayer16": {
        "group_values": (0, 0, 0, 5, 6),
        "excess_card_value": 0,
        "commitment_per_investment": Fraction(3, 2),
    },
    # Four-card specialist: strong immediate reward for four identical cards.
    # This is intentionally shortsighted: the compulsory play breaks that group.
    "RankingPlayer17": {
        "group_values": (0, 0, 0, 1, 8),
        "excess_card_value": 0,
        "commitment_per_investment": 2,
        "new_token_value": -1,
    },
    # Protective investor: strongly resist risky offers while tolerating tokens;
    # somewhat stronger minority penalties and lead rewards support that bias.
    "RankingPlayer18": {
        "offer_risk_per_card": -4,
        "majority_values": (-5, -3, 0, 1, 4),
        "new_token_value": -1,
    },
    # Surplus trimmer: strongly discourage excess combined holdings and further
    # dominant investment; weaker groups/commitment reduce incentives to hoard.
    "RankingPlayer19": {
        "excess_card_value": -3,
        "dominated_investment_value": -4,
        "group_values": (0, 0, 1, 2, 3),
        "commitment_per_investment": Fraction(1, 2),
    },
}


def create_ranking_player(
    player_id: PlayerId, config_name: str = "RankingPlayer1", *,
    seed: int | None = None, name: str | None = None,
) -> WeightedRankingPlayer:
    """Construct a fresh controller from a named dictionary, optionally renamed.

    Validate the configuration name and pass its overrides to the shared
    evaluator. The default display name identifies the chosen configuration.
    Duplicate selections produce independent controllers/RNGs, not singletons.
    """
    if not isinstance(config_name, str) or config_name not in RANKING_CONFIGS:
        raise ValueError(f"Unknown ranking configuration: {config_name!r}")
    return WeightedRankingPlayer(
        player_id, f"Ranking ({config_name})" if name is None else name,
        seed=seed, weights=RANKING_CONFIGS[config_name],
    )
