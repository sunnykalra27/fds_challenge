import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from collections import defaultdict, Counter
TYPE_CHART = {
    'fire': {'grass': 2, 'water': 0.5, 'rock': 0.5, 'ice': 2},
    'water': {'fire': 2, 'grass': 0.5, 'rock': 2, 'ground': 2},
    'grass': {'water': 2, 'fire': 0.5, 'rock': 2, 'ground': 2},
    'electric': {'water': 2, 'ground': 0, 'flying': 2, 'grass': 0.5},
    'normal': {'rock': 0.5, 'ghost': 0},
}

def get_type_advantage(p1_types, p2_types):
    """Compute average type advantage score for Player 1's team vs Player 2's lead."""
    if not p1_types or not p2_types:
        return 1.0
    scores = []
    for t1 in p1_types:
        for t2 in p2_types:
            t1, t2 = t1.lower(), t2.lower()
            scores.append(TYPE_CHART.get(t1, {}).get(t2, 1.0))
    return np.mean(scores) if scores else 1.0

def create_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    for battle in tqdm(data, desc="Extracting features"):
        f = {}
        f["battle_id"] = battle.get("battle_id")
        f["player_won"] = battle.get("player_won", 0)

        # --- Player 1 team ---
        p1_team = battle.get("p1_team_details", [])
        if p1_team:
            f["p1_hp_mean"]  = np.mean([p.get("base_hp", 0) for p in p1_team])
            f["p1_atk_mean"] = np.mean([p.get("base_atk", 0) for p in p1_team])
            f["p1_def_mean"] = np.mean([p.get("base_def", 0) for p in p1_team])
            f["p1_spe_mean"] = np.mean([p.get("base_spe", 0) for p in p1_team])

            # --- Collect all types (flatten and lowercase) ---
            p1_types = []
            for p in p1_team:
                for t in p.get("types", []):
                    if isinstance(t, str):
                        p1_types.append(t.lower())
            f["p1_type_diversity"] = len(set(p1_types))
        else:
            p1_types = []
            f.update({"p1_hp_mean":0,"p1_atk_mean":0,"p1_def_mean":0,"p1_spe_mean":0,"p1_type_diversity":0})

        # --- Player 2 lead ---
        p2_lead = battle.get("p2_lead_details", {})
        f["p2_hp"]  = p2_lead.get("base_hp", 0)
        f["p2_atk"] = p2_lead.get("base_atk", 0)
        f["p2_def"] = p2_lead.get("base_def", 0)
        f["p2_spe"] = p2_lead.get("base_spe", 0)
        p2_types = [t.lower() for t in p2_lead.get("types", ["notype"]) if isinstance(t, str)]

        # --- Type advantage ---
        f["type_advantage"] = get_type_advantage(p1_types, p2_types)

        # --- Stat differences ---
        for stat in ["hp", "atk", "def", "spe"]:
            f[f"{stat}_diff"]  = f.get(f"p1_{stat}_mean", 0) - f.get(f"p2_{stat}", 0)
            f[f"{stat}_ratio"] = f.get(f"p1_{stat}_mean", 0) / (f.get(f"p2_{stat}", 0) + 1e-6)

        # --- Timeline features ---
        timeline = battle.get("battle_timeline", [])
        if timeline:
            p1_hp = np.array([t.get("p1_hp", 0) for t in timeline])
            p2_hp = np.array([t.get("p2_hp", 0) for t in timeline])
            hp_diff = p1_hp - p2_hp
            f["num_turns"] = len(timeline)
            f["hp_diff_mean"] = np.mean(hp_diff)
            f["hp_diff_std"] = np.std(hp_diff)
            f["hp_diff_final"] = hp_diff[-1]
            turns = np.arange(len(hp_diff))
            f["hp_trend_slope"] = np.polyfit(turns, hp_diff, 1)[0] if len(turns) > 1 else 0
        else:
            f.update({"num_turns":0,"hp_diff_mean":0,"hp_diff_std":0,"hp_diff_final":0,"hp_trend_slope":0})
                # --- Battle Momentum Features ---
        if timeline and len(timeline) > 2:
            p1_hp = np.array([t.get("p1_hp", 0) for t in timeline])
            p2_hp = np.array([t.get("p2_hp", 0) for t in timeline])
            hp_diff = p1_hp - p2_hp

            # Who was leading at each turn (+1 if P1 ahead, -1 if P2 ahead)
            lead_sign = np.sign(hp_diff)
            lead_changes = np.sum(np.diff(lead_sign) != 0)

            # Did P1 keep the lead from start to end?
            f["first_lead_retained"] = 1 if (lead_sign[0] > 0 and lead_sign[-1] > 0) else 0

            # How often P1 was ahead overall
            f["lead_fraction"] = np.mean(lead_sign > 0)

            # Momentum = average change in HP diff per turn
            f["hp_momentum"] = (hp_diff[-1] - hp_diff[0]) / len(hp_diff)

            # Variability = how swingy the battle was
            f["hp_variability"] = np.std(np.diff(hp_diff))

            # Total lead changes
            f["lead_changes"] = lead_changes
        else:
            f.update({
                "first_lead_retained": 0,
                "lead_fraction": 0,
                "hp_momentum": 0,
                "hp_variability": 0,
                "lead_changes": 0
            })

                # --- Derived Interaction Features ---
        f["adv_speed_interaction"] = f["type_advantage"] * f["spe_diff"]
        f["adv_attack_interaction"] = f["type_advantage"] * f["atk_ratio"]
        f["momentum_advantage"] = f["hp_momentum"] * f["type_advantage"]
        f["stat_balance"] = abs(f["atk_ratio"] - f["def_ratio"])
        f["battle_dominance"] = (f["hp_diff_final"] + f["hp_diff_mean"]) / (f["num_turns"] + 1)

                # --- Advanced Timeline Trend Features ---
        if timeline and len(timeline) > 3:
            p1_hp = np.array([t.get("p1_hp", 0) for t in timeline])
            p2_hp = np.array([t.get("p2_hp", 0) for t in timeline])
            hp_diff = p1_hp - p2_hp
            turns = np.arange(len(hp_diff))

            # Normalize HP difference per turn (for stability)
            norm_hp_diff = hp_diff / (np.max(np.abs(hp_diff)) + 1e-6)

            # Linear regression slope (how P1 advantage changes per turn)
            f["hp_diff_slope"] = np.polyfit(turns, hp_diff, 1)[0]

            # Relative trend (how the difference evolves compared to average)
            f["hp_trend_strength"] = f["hp_diff_slope"] / (np.std(hp_diff) + 1e-6)

            # Average HP ratio per turn
            hp_ratio = p1_hp / (p2_hp + 1e-6)
            f["avg_hp_ratio"] = np.mean(hp_ratio)

            # Stability of lead (how consistent advantage is)
            f["lead_stability"] = 1 - (np.std(norm_hp_diff))

            # Did P1 recover from early disadvantage?
            f["recovery_flag"] = 1 if (hp_diff[0] < 0 and hp_diff[-1] > 0) else 0
        else:
            f.update({
                "hp_diff_slope": 0,
                "hp_trend_strength": 0,
                "avg_hp_ratio": 0,
                "lead_stability": 0,
                "recovery_flag": 0
            })


        feature_list.append(f)

    return pd.DataFrame(feature_list).fillna(0)

BASIC_STATS = ["hp","atk","def","spa","spd","spe"]

def build_features(battles):
    rows = []

    for battle in battles:
        features = {}

        me_team = battle.get("p1_team_details", []) or []
        if me_team:
            for stat in BASIC_STATS:
                values = [p.get(f"base_{stat}", 0) for p in me_team]
                features[f"me_average_{stat}"] = float(np.mean(values))
                features[f"me_total_{stat}"] = float(np.sum(values))
                features[f"me_maximum_{stat}"] = float(np.max(values))

            features["me_fast_pokemon_count"] = int(sum(p.get("base_spe",0) >= 100 for p in me_team))
            features["me_high_hp_pokemon_count"] = int(sum(p.get("base_hp",0) >= 90 for p in me_team))
            features["me_high_special_attack_pokemon_count"] = int(sum(p.get("base_spa",0) >= 110 for p in me_team))
        else:
            for stat in BASIC_STATS:
                for agg in ["average","total","maximum"]:
                    features[f"me_{agg}_{stat}"] = 0.0
            features["me_fast_pokemon_count"] = 0
            features["me_high_hp_pokemon_count"] = 0
            features["me_high_special_attack_pokemon_count"] = 0

        opponent_lead = battle.get("p2_lead_details", {}) or {}
        for stat in BASIC_STATS:
            features[f"opponent_{stat}"] = float(opponent_lead.get(f"base_{stat}", 0))

        for stat in BASIC_STATS:
            features[f"average_{stat}_difference"] = (features.get(f"me_average_{stat}", 0.0) - features.get(f"opponent_{stat}", 0.0))

        timeline = (battle.get("battle_timeline", []) or [])[:30]

        me_total_damage_done = 0.0
        opponent_total_damage_done = 0.0
        me_attack_move_count = me_status_move_count = 0
        opponent_attack_move_count = opponent_status_move_count = 0
        me_hp_series, opponent_hp_series = [], []
        previous_me_hp = previous_opponent_hp = None

        for event in timeline:
            me_state = event.get("p1_pokemon_state", {}) or {}
            opponent_state = event.get("p2_pokemon_state", {}) or {}

            me_hp = me_state.get("hp_pct")
            opponent_hp = opponent_state.get("hp_pct")

            if me_hp is not None: me_hp_series.append(me_hp)
            if opponent_hp is not None: opponent_hp_series.append(opponent_hp)

            if previous_me_hp is not None and me_hp is not None:
                drop = previous_me_hp - me_hp
                if drop > 0:
                    opponent_total_damage_done += drop
            if previous_opponent_hp is not None and opponent_hp is not None:
                drop = previous_opponent_hp - opponent_hp
                if drop > 0:
                    me_total_damage_done += drop

            if me_hp is not None: previous_me_hp = me_hp
            if opponent_hp is not None: previous_opponent_hp = opponent_hp

            me_move = event.get("p1_move_details") or {}
            opponent_move = event.get("p2_move_details") or {}

            if me_move:
                if me_move.get("category") == "STATUS":
                    me_status_move_count += 1
                else:
                    me_attack_move_count += 1
            if opponent_move:
                if opponent_move.get("category") == "STATUS":
                    opponent_status_move_count += 1
                else:
                    opponent_attack_move_count += 1

        features["me_total_damage_done"] = me_total_damage_done
        features["opponent_total_damage_done"] = opponent_total_damage_done
        features["me_damage_difference"] = me_total_damage_done - opponent_total_damage_done

        if me_hp_series:
            features["me_min_hp_percent"] = float(min(me_hp_series))
            features["me_average_hp_percent"] = float(np.mean(me_hp_series))
        else:
            features["me_min_hp_percent"] = 1.0
            features["me_average_hp_percent"] = 1.0

        if opponent_hp_series:
            features["opponent_min_hp_percent"] = float(min(opponent_hp_series))
            features["opponent_average_hp_percent"] = float(np.mean(opponent_hp_series))
        else:
            features["opponent_min_hp_percent"] = 1.0
            features["opponent_average_hp_percent"] = 1.0

        features["me_attack_move_count"] = me_attack_move_count
        features["me_status_move_count"] = me_status_move_count
        features["opponent_attack_move_count"] = opponent_attack_move_count
        features["opponent_status_move_count"] = opponent_status_move_count

        features["battle_id"] = battle.get("battle_id")
        if "player_won" in battle:
            features["player_won"] = int(battle["player_won"])

        rows.append(features)

    return pd.DataFrame(rows).fillna(0.0) #make Nan ==0
