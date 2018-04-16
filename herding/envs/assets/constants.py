import json
from enum import Enum


class SheepType(Enum):
    SIMPLE, COMPLEX, CONTINUOUS = range(3)


class RotationMode(Enum):
    FREE, LOCKED_ON_HERD_CENTRE = range(2)


class AgentLayout(Enum):
    RANDOM, LAYOUT1, LAYOUT2 = range(3)


class AgentObservationRepresentation(Enum):
    COMPRESSED, TWO_CHANNEL, TWO_CHANNEL_FLATTENED, TARGET_TYPE_ONLY = range(4)


class AgentObservationAids(Enum):
    NO, COMPASS, TO_MASS_CENTER = range(3)


class RewardCalculatorType(Enum):
    SCATTER_DIFFERENCE, IN_TARGET_DIFFERENCE, COMPLEX = range(3)


# below is the code to serialize enums into json
# taken from: https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json

PUBLIC_ENUMS = {
    'SheepType': SheepType,
    'RotationMode': RotationMode,
    'AgentLayout': AgentLayout,
    'AgentObservationRepresentation': AgentObservationRepresentation,
    'AgentObservationAids': AgentObservationAids,
    'RewardCalculatorType': RewardCalculatorType
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d
