
def reinforce_vow(vow, method):
    if method not in vow["reinforced_by"]:
        vow["reinforced_by"].append(method)
    vow["devotion_level"] = min(1.0, round(vow["devotion_level"] + 0.05, 4))

def decay_vows(vow):
    vow["devotion_level"] = max(0.0, round(vow["devotion_level"] - 0.01, 4))
