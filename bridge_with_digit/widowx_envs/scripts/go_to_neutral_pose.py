#!/usr/bin/env python3

if __name__ == '__main__':
    from widowx_envs.widowx_env import StateReachingWidowX
    env = StateReachingWidowX()
    env.move_to_neutral()
