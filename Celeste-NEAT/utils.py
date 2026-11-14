from math import pi

from CelestePythonInterface import SessionData
from CelestePythonInterface.SessionData import (
    DISTANCE_END,
    DISTANCE_START,
    INPUT_VALUE_COUNT,
    LOCAL_GRID_END,
    LOCAL_GRID_START,
    OBJECT_TYPE_END,
    OBJECT_TYPE_START,
)
from llm_planner import PLAN_FEATURE_COUNT


LEGACY_RAY_INDICES = [0, 4, 8, 12, 16, 20, 24, 28]
DEFAULT_INPUT_COUNT = INPUT_VALUE_COUNT + PLAN_FEATURE_COUNT


def safe_div(value, divisor):
    if abs(divisor) < 0.0001:
        return 0.0
    return value / divisor


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def legacy_inputs_preprocessor(player_state):
    level_diagonal = player_state[SessionData.LEVEL_DIAGONAL_LENGTH]
    inputs = []

    for index in LEGACY_RAY_INDICES:
        inputs.append(clamp(safe_div(player_state[DISTANCE_START + index], level_diagonal), 0.0, 1.0))

    for index in LEGACY_RAY_INDICES:
        object_type = player_state[OBJECT_TYPE_START + index]
        inputs.append(0.0 if object_type == 0 or object_type == 3 else 1.0)

    inputs.extend(
        [
            clamp(player_state[SessionData.X_VELOCITY] / 300.0),
            clamp(player_state[SessionData.Y_VELOCITY] / 300.0),
            player_state[SessionData.CAN_DASH],
            player_state[SessionData.ON_GROUND],
            clamp(player_state[SessionData.STAMINA] / 110.0, 0.0, 1.0),
            safe_div(player_state[SessionData.ANGLE_TO_OBJECTIVE], 2 * pi),
            clamp(safe_div(player_state[SessionData.DISTANCE_TO_OBJECTIVE], level_diagonal), 0.0, 1.0),
        ]
    )
    return inputs


def enhanced_inputs_preprocessor(player_state):
    inputs = list(player_state[:INPUT_VALUE_COUNT])
    level_diagonal = player_state[SessionData.LEVEL_DIAGONAL_LENGTH]

    for i in range(DISTANCE_START, DISTANCE_END):
        inputs[i] = clamp(safe_div(inputs[i], level_diagonal), 0.0, 1.0)

    for i in range(OBJECT_TYPE_START, OBJECT_TYPE_END):
        inputs[i] = clamp(inputs[i] / 3.0, 0.0, 1.0)

    for i in range(LOCAL_GRID_START, LOCAL_GRID_END):
        inputs[i] = clamp(inputs[i] / 3.0, 0.0, 1.0)

    inputs[SessionData.X_VELOCITY] = clamp(inputs[SessionData.X_VELOCITY] / 300.0)
    inputs[SessionData.Y_VELOCITY] = clamp(inputs[SessionData.Y_VELOCITY] / 300.0)
    inputs[SessionData.STAMINA] = clamp(inputs[SessionData.STAMINA] / 110.0, 0.0, 1.0)
    inputs[SessionData.ANGLE_TO_OBJECTIVE] = safe_div(inputs[SessionData.ANGLE_TO_OBJECTIVE], 2 * pi)
    inputs[SessionData.DISTANCE_TO_OBJECTIVE] = clamp(safe_div(inputs[SessionData.DISTANCE_TO_OBJECTIVE], level_diagonal), 0.0, 1.0)
    inputs[SessionData.INITIAL_DISTANCE_TO_OBJECTIVE] = clamp(safe_div(inputs[SessionData.INITIAL_DISTANCE_TO_OBJECTIVE], level_diagonal), 0.0, 1.0)
    inputs[SessionData.BEST_DISTANCE_TO_OBJECTIVE] = clamp(safe_div(inputs[SessionData.BEST_DISTANCE_TO_OBJECTIVE], level_diagonal), 0.0, 1.0)
    inputs[SessionData.PROGRESS_TO_OBJECTIVE] = clamp(safe_div(inputs[SessionData.PROGRESS_TO_OBJECTIVE], level_diagonal))
    inputs[SessionData.DELTA_X_TO_OBJECTIVE] = clamp(inputs[SessionData.DELTA_X_TO_OBJECTIVE])
    inputs[SessionData.DELTA_Y_TO_OBJECTIVE] = clamp(inputs[SessionData.DELTA_Y_TO_OBJECTIVE])
    inputs[SessionData.HORIZONTAL_PROGRESS] = clamp(safe_div(inputs[SessionData.HORIZONTAL_PROGRESS], level_diagonal))
    inputs[SessionData.VERTICAL_PROGRESS] = clamp(safe_div(inputs[SessionData.VERTICAL_PROGRESS], level_diagonal))
    inputs[SessionData.SPEED_MAGNITUDE] = clamp(inputs[SessionData.SPEED_MAGNITUDE] / 300.0, 0.0, 1.0)

    return inputs


def inputs_preprocessor(player_state, input_count=DEFAULT_INPUT_COUNT, room_plan=None, route_planner=None):
    if input_count <= 23:
        return legacy_inputs_preprocessor(player_state)[:input_count]

    inputs = enhanced_inputs_preprocessor(player_state)
    if input_count > INPUT_VALUE_COUNT:
        if route_planner is not None and room_plan is not None:
            inputs.extend(route_planner.features_for_state(player_state, room_plan))
        else:
            inputs.extend([0.0] * PLAN_FEATURE_COUNT)
    return inputs[:input_count]


def score_function(final_player_state, session_parameters):
    finished = final_player_state[SessionData.NUMBER_OF_LEVELS_FINISHED] >= 1
    died = final_player_state[SessionData.DIED] >= 0.5
    timeout = float(session_parameters.TimeoutSeconds)
    seconds_elapsed = final_player_state[SessionData.SECONDS_ELAPSED]

    initial_distance = final_player_state[SessionData.INITIAL_DISTANCE_TO_OBJECTIVE]
    final_distance = final_player_state[SessionData.DISTANCE_TO_OBJECTIVE]
    best_distance = final_player_state[SessionData.BEST_DISTANCE_TO_OBJECTIVE]
    progress = initial_distance - final_distance
    best_progress = initial_distance - best_distance

    fitness = 0.0
    fitness += best_progress * 12.0
    fitness += progress * 5.0
    fitness -= final_distance * 0.15
    fitness += final_player_state[SessionData.HORIZONTAL_PROGRESS] * 0.75
    fitness += final_player_state[SessionData.VERTICAL_PROGRESS] * 0.25
    fitness += min(seconds_elapsed, timeout) * 2.0

    if final_player_state[SessionData.IS_STUCK] >= 0.5:
        fitness -= 250.0

    if died:
        fitness -= 750.0

    if finished:
        fitness += 10000.0
        fitness += max(0.0, timeout - seconds_elapsed) * 1000.0

    return fitness
