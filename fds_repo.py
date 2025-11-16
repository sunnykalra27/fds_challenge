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

SEED = 42
np.random.seed(SEED)
BASIC_STATS = ["hp","atk","def","spa","spd","spe"]

TYPES = [
    'dragon','electric','fire','flying','ghost','grass',
    'ground','ice','normal','notype','poison','psychic','rock','water']

def get_types(obj):
    """Extract types list from a pokemon dict (fallback to 'notype')."""
    if not isinstance(obj, dict): return []
    ts = obj.get("types", [])
    if not isinstance(ts, (list, tuple)): return []
    out = []
    for t in ts:
        if isinstance(t, str):
            out.append(t.lower())
        else:
            out.append("notype")
    return out

def learn_type_effectiveness(battles, min_battles=5):
    type_win_count  = defaultdict(lambda: defaultdict(int))
    type_battle_cnt = defaultdict(lambda: defaultdict(int))

    for b in battles:
        p1_team = b.get("p1_team_details", []) or []
        p2_lead = b.get("p2_lead_details", {}) or {}
        if not p1_team or not p2_lead: 
            continue

        p2_types = get_types(p2_lead) or ["notype"]
        for p1 in p1_team:
            p1_types = get_types(p1) or ["notype"]
            for a in p1_types:
                for d in p2_types:
                    type_battle_cnt[a][d] += 1
                    if b.get("player_won") == 1:  
                        type_win_count[a][d] += 1
                    type_battle_cnt[d][a] += 1
                    if b.get("player_won") == 0:  
                        type_win_count[d][a] += 1

    eff = defaultdict(lambda: defaultdict(lambda: 1.0))
    for a in type_battle_cnt:
        for d in type_battle_cnt[a]:
            n = type_battle_cnt[a][d]
            if n < min_battles:
                eff[a][d] = 1.0
            else:
                win_rate = type_win_count[a][d] / n
                eff[a][d] = max(0.5, min(1.5, 0.5 + win_rate)) 
    return eff



def type_matchup_score(attacker_types, defender_types, eff_table):
    if not attacker_types or not defender_types:
        return 1.0
    scores = []
    for at in attacker_types:
        mults = [eff_table[at].get(dt, 1.0) for dt in defender_types]
        scores.append(max(mults) if mults else 1.0)
    return float(np.mean(scores)) if scores else 1.0

def summarize_timeline_30(battle, max_turns=30):
    tl = (battle.get("battle_timeline", []) or [])[:max_turns]
    if not tl:
        return {k:0.0 for k in [
            "n_turns","p1_switches","p2_switches","p1_status_count","p2_status_count",
            "p1_avg_hp_pct","p2_avg_hp_pct","p1_damage_per_turn","p2_damage_per_turn",
            "p1_boost_sum","p2_boost_sum","p1_turns_leading","p1_moves_count","p2_moves_count",
            "p1_lead_first10","lead_changes","p1_last5_damage","p2_last5_damage","hp_diff_mean"]}

    def approx_damage(hplist):
        if len(hplist) < 2: return 0.0
        arr = np.array(hplist, dtype=float)
        diffs = np.maximum(0, arr[:-1] - arr[1:])
        return float(diffs.mean()) if len(diffs) else 0.0

    def last_k_damage(hplist, k=5):
        if len(hplist) < 2: return 0.0
        arr = np.array(hplist, dtype=float)
        diffs = np.maximum(0, arr[:-1] - arr[1:])
        tail = diffs[-k:] if len(diffs) >= k else diffs
        return float(tail.mean()) if len(tail) else 0.0

    p1_hp, p2_hp = [], []
    p1_switches = p2_switches = 0
    p1_status = p2_status = 0
    p1_boost = p2_boost = 0.0
    p1_moves = p2_moves = 0
    prev_p1_name = prev_p2_name = None
    lead_flags = []
    p1_lead_first10 = 0

    prev_p1_hp = prev_p2_hp = None

    for idx, ev in enumerate(tl, start=1):
        p1s = ev.get("p1_pokemon_state") or {}
        p2s = ev.get("p2_pokemon_state") or {}

        n1 = p1s.get("name")
        if prev_p1_name and n1 and n1 != prev_p1_name:
            p1_switches += 1
        prev_p1_name = n1 or prev_p1_name

        n2 = p2s.get("name")
        if prev_p2_name and n2 and n2 != prev_p2_name:
            p2_switches += 1
        prev_p2_name = n2 or prev_p2_name

        h1 = p1s.get("hp_pct", 0.0); p1_hp.append(h1)
        h2 = p2s.get("hp_pct", 0.0); p2_hp.append(h2)

        if p1s.get("status") not in [None, "nostatus"]: p1_status += 1
        if p1s.get("main_status") not in [None, "nostatus"]: p1_status += 1
        if p2s.get("status") not in [None, "nostatus"]: p2_status += 1
        if p2s.get("main_status") not in [None, "nostatus"]: p2_status += 1

        b1 = p1s.get("boosts") or {}; p1_boost += sum(max(0, v) for v in b1.values())
        b2 = p2s.get("boosts") or {}; p2_boost += sum(max(0, v) for v in b2.values())

        if ev.get("p1_move_details"): p1_moves += 1
        if ev.get("p2_move_details"): p2_moves += 1

        if p1s and p2s:
            lead = (p1s.get("hp_pct", 0) > p2s.get("hp_pct", 0))
            lead_flags.append(lead)
            if lead and idx <= 10: p1_lead_first10 += 1

    lead_changes = 0
    if len(lead_flags) >= 2:
        lf = np.array(lead_flags, dtype=int)
        lead_changes = int(np.sum(lf[1:] != lf[:-1]))

    out = {
        "n_turns": len(tl),
        "p1_switches": p1_switches, "p2_switches": p2_switches,
        "p1_status_count": p1_status, "p2_status_count": p2_status,
        "p1_avg_hp_pct": float(np.mean(p1_hp)) if p1_hp else 0.0,
        "p2_avg_hp_pct": float(np.mean(p2_hp)) if p2_hp else 0.0,
        "p1_damage_per_turn": approx_damage(p1_hp),
        "p2_damage_per_turn": approx_damage(p2_hp),
        "p1_boost_sum": p1_boost, "p2_boost_sum": p2_boost,
        "p1_turns_leading": int(sum(lead_flags)) if lead_flags else 0,
        "p1_moves_count": p1_moves, "p2_moves_count": p2_moves,
        "p1_lead_first10": p1_lead_first10,
        "lead_changes": lead_changes,
        "p1_last5_damage": last_k_damage(p1_hp, 5),
        "p2_last5_damage": last_k_damage(p2_hp, 5),
        "hp_diff_mean": (float(np.mean(p1_hp)) - float(np.mean(p2_hp))) if (p1_hp and p2_hp) else 0.0}
    return out

def build_enhanced_features(battles, TYPE_EFF):
    rows = []

    for b in battles:
        f = {}

        me_team = b.get("p1_team_details", []) or []

        if me_team:
            for stat in BASIC_STATS:
                vals = [p.get(f"base_{stat}", 0) for p in me_team]
                f[f"me_average_{stat}"] = float(np.mean(vals))
                f[f"me_total_{stat}"]   = float(np.sum(vals))
                f[f"me_maximum_{stat}"] = float(np.max(vals))
            f["me_fast_pokemon_count"]              = int(sum(p.get("base_spe",0) >= 100 for p in me_team))
            f["me_high_hp_pokemon_count"]           = int(sum(p.get("base_hp",0)  >=  90 for p in me_team))
            f["me_high_special_attack_pokemon_count"]= int(sum(p.get("base_spa",0) >= 110 for p in me_team))
        else:
            for stat in BASIC_STATS:
                f[f"me_average_{stat}"]=0.0; f[f"me_total_{stat}"]=0.0; f[f"me_maximum_{stat}"]=0.0
            f["me_fast_pokemon_count"]=0; f["me_high_hp_pokemon_count"]=0; f["me_high_special_attack_pokemon_count"]=0

        opp_lead = b.get("p2_lead_details", {}) or {}
        for stat in BASIC_STATS:
            f[f"opponent_{stat}"] = float(opp_lead.get(f"base_{stat}", 0))
        for stat in BASIC_STATS:
            f[f"average_{stat}_difference"] = f.get(f"me_average_{stat}",0.0) - f.get(f"opponent_{stat}",0.0)

        f.update(summarize_timeline_30(b, max_turns=30))

        p2_types = get_types(opp_lead) or ["notype"]
        team_type_scores = []
        for p in me_team:
            tts = get_types(p) or ["notype"]
            team_type_scores.append(type_matchup_score(tts, p2_types, TYPE_EFF))
        if not team_type_scores: team_type_scores = [1.0]
        f["p1_vs_p2lead_type_mean"] = float(np.mean(team_type_scores))
        f["p1_vs_p2lead_type_max"]  = float(np.max(team_type_scores))
        f["p1_vs_p2lead_type_min"]  = float(np.min(team_type_scores))

        lead_vs_team = []
        for p in me_team:
            tts = get_types(p) or ["notype"]
            lead_vs_team.append(type_matchup_score(p2_types, tts, TYPE_EFF))
        if not lead_vs_team: lead_vs_team = [1.0]
        f["p2lead_vs_p1team_type_mean"] = float(np.mean(lead_vs_team))

        f["damage_balance"]           = f.get("p1_damage_per_turn",0.0) - f.get("p2_damage_per_turn",0.0)
        f["finishing_pressure"]       = f.get("p1_last5_damage",0.0)    - f.get("p2_last5_damage",0.0)
        n_first = min(10, int(f.get("n_turns",0)))
        f["early_control_ratio"]      = (f.get("p1_lead_first10",0)/n_first) if n_first>0 else 0.0
        n_turns = max(1, int(f.get("n_turns",1)))
        f["lead_volatility"]          = f.get("lead_changes",0)/n_turns
        p1_total = sum([f.get(f"me_average_{s}",0.0) for s in BASIC_STATS])
        p2_total = sum([f.get(f"opponent_{s}",0.0)   for s in BASIC_STATS])
        f["stat_total_diff"]          = p1_total - p2_total
        p1_moves = max(1, int(f.get("p1_moves_count",0)))
        p2_moves = max(1, int(f.get("p2_moves_count",0)))
        p1_eff = f.get("p1_damage_per_turn",0.0)/p1_moves
        p2_eff = f.get("p2_damage_per_turn",0.0)/p2_moves
        f["offensive_efficiency_diff"] = p1_eff - p2_eff
        f["control_index"] = (
            0.6 * f["damage_balance"]
          + 0.4 * f["finishing_pressure"]
          + 0.5 * f.get("hp_diff_mean",0.0)
          - 0.3 * (f.get("p1_switches",0) - f.get("p2_switches",0))
        )
        f["momentum_hp"] = (f.get("p1_avg_hp_pct",0.0) - f.get("p2_avg_hp_pct",0.0)) * f.get("n_turns",1)
        f["damage_balance_norm"] = f["damage_balance"] / n_turns
        f["boost_diff_norm"]     = (f.get("p1_boost_sum",0.0) - f.get("p2_boost_sum",0.0)) / n_turns

        # --- ids/label (kept) ---
        f["battle_id"] = b.get("battle_id")
        if "player_won" in b:
            f["player_won"] = int(b["player_won"])

        rows.append(f)

    return pd.DataFrame(rows).fillna(0.0)


