#!/usr/bin/python3

import balltracker

if __name__ == "__main__":
    from IPython import embed

    ball_tracker = balltracker.BallTracker()
    ball_tracker._init_("./videos/")

    embed()
