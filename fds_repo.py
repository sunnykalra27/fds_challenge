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
