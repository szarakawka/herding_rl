class SheepType:
    SIMPLE, COMPLEX, CONTINUOUS = range(3)


class RotationMode:
    FREE, LOCKED_ON_HERD_CENTRE = range(2)


class AgentLayout:
    RANDOM, LAYOUT1, LAYOUT2 = range(3)


class AgentObservationCompression:
    COMPRESSED, TWO_CHANNEL = range(2)


class AgentObservationAids:
    NO, COMPASS, TO_MASS_CENTER = range(3)
