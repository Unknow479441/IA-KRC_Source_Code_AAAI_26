from numpy.random import RandomState
from os.path import dirname, join
from functools import partial
from itertools import combinations_with_replacement, product


def get_all_unique_teams(all_types, min_len, max_len):
    all_uniq = []
    for i in range(min_len, max_len + 1):
        all_uniq += list(combinations_with_replacement(all_types, i))
    all_uniq_counts = []
    for scen in all_uniq:
        curr_uniq = list(set(scen))
        uniq_counts = list(zip([scen.count(u) for u in curr_uniq], curr_uniq))
        all_uniq_counts.append(uniq_counts)
    return all_uniq_counts


def fixed_armies(ally_army, enemy_army, ally_centered=False, rotate=False,
                 separation=10, jitter=0, episode_limit=100,
                 map_name="empty_passive", n_extra_tags=0, rs=None):
    # 添加调试输出
    
    # 为max_types_and_units_scenario创建不含位置的版本
    ally_units = [(num, unit) for num, unit, _ in ally_army]
    enemy_units = [(num, unit) for num, unit, _ in enemy_army]
    
    scenario_dict = {
        # 直接使用包含位置信息的场景
        'scenarios': [(ally_army, enemy_army)],
        'max_types_and_units_scenario': (ally_units, enemy_units),
        'ally_centered': ally_centered,
        'rotate': rotate,
        'separation': separation,
        'jitter': jitter,
        'episode_limit': episode_limit,
        'n_extra_tags': n_extra_tags,
        'map_name': map_name
    }
    return scenario_dict


def symmetric_armies(army_spec, ally_centered=False,
                     rotate=False, separation=10,
                     jitter=0, episode_limit=100, map_name="empty_passive",
                     n_extra_tags=0,
                     rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    unique_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]

    scenarios = list(zip(unique_teams, unique_teams))
    # sort by number of types and total number of units
    max_types_and_units_team = sorted(unique_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_team,
                                    max_types_and_units_team)

    scenario_dict = {'scenarios': scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'n_extra_tags': n_extra_tags,
                     'map_name': map_name}
    return scenario_dict


def asymm_armies(army_spec, spec_delta, ally_centered=False,
                 rotate=False, separation=10,
                 jitter=0, episode_limit=100, map_name="empty_passive",
                 n_extra_tags=0,
                 rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    enemy_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]
    agent_teams = [[(max(num + spec_delta.get(typ, 0), 0), typ) for num, typ in team] for team in enemy_teams]

    scenarios = list(zip(agent_teams, enemy_teams))
    # sort by number of types and total number of units
    max_types_and_units_ag_team = sorted(agent_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_en_team = sorted(enemy_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_ag_team,
                                    max_types_and_units_en_team)

    scenario_dict = {'scenarios': scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'n_extra_tags': n_extra_tags,
                     'map_name': map_name}
    return scenario_dict



custom_scenario_registry = {
  "3m_fixed": partial(fixed_armies,
    ally_army=[
        (1, "Marine", (-6.5, -0.5)),
        (1, "Marine", (-6.5, 3.5)),
        (1, "Marine", (-6.5, -4.5)),
        (1, "Marine", (-9.5, 1.5)),
        (1, "Marine", (-9.5, -2.5))
    ],
    enemy_army=[
        (1, "Marine", (5.5, -0.5)),
        (1, "Marine", (5.5, 3.5)),
        (1, "Marine", (5.5, -4.5)),
        (1, "Marine", (8.5, 1.5)),
        (1, "Marine", (8.5, -2.5))
    ],    ally_centered=False,
    rotate=False,
    separation=0,
    jitter=0,
    episode_limit=300,
    map_name="empty_passive"
  ),
  "3-8m_symmetric": partial(symmetric_armies,
                            [(('Marine',), (3, 8))],
                            rotate=True,
                            ally_centered=False,
                            separation=14,
                            jitter=1, episode_limit=150, map_name="empty_passive"),
  "6-11m_mandown": partial(asymm_armies,
                          [(('Marine',), (6, 11))],
                          {'Marine': -1},
                          rotate=True,
                          ally_centered=False,
                          separation=14,
                          jitter=1, episode_limit=100, map_name="empty_passive"),
  "3-8sz_symmetric": partial(symmetric_armies,
                             [(('Stalker', 'Zealot'), (3, 8))],
                             rotate=True,
                             ally_centered=False,
                             separation=14,
                             jitter=3, episode_limit=150, map_name="empty_passive"),
  "3-8MMM_symmetric": partial(symmetric_armies,
                              [(('Marine', 'Marauder'), (3, 6)),
                               (('Medivac',), (0, 2))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=7, episode_limit=500, map_name="empty_passive"),
  "3-8csz_symmetric": partial(symmetric_armies,
                              [(('Stalker', 'Zealot'), (3, 6)),
                               (('Colossus',), (0, 2))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=1, episode_limit=150, map_name="empty_passive"),
}
