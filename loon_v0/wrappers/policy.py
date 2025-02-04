class MyPolicy:
    def __init__(self, action_space, targ_altitude=3000):
        self.action_space = action_space
        self.targ_altitude = targ_altitude

    def get_action(self, observation):
        altitude = observation["altitude"]

        if altitude < self.targ_altitude:     # If too low, increase altitude
            diff = self.targ_altitude - altitude
            return (1, diff)
        elif altitude > self.targ_altitude:   # If too high, decrease altitude
            return (2, altitude - self.targ_altitude)
        else:
            return (0, 0)    # Otherwise, do nothing
