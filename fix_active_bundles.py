#!/usr/bin/env python3
import os
import json
from shared_utils import load_state, save_state, required_platforms


def fix_state(state_path):
    if not os.path.exists(state_path):
        return
    
    state = load_state(state_path)
    active = state.get('active_bundle')
    if active and isinstance(active, dict):
        required = required_platforms(state_path)
        posted = active.get('platforms_posted', [])
        missing = [p for p in required if p not in posted]
        post_id = active.get('post_id', 'N/A')
        print(f"Checking {state_path}:")
        print(f"  Active bundle Post ID: {post_id}")
        print(f"  Posted so far: {posted}")
        print(f"  Required: {required}")
        print(f"  Still missing: {missing or 'none'}")

        if not missing:
            print("  All required platforms posted! Clearing active_bundle...")
            state['active_bundle'] = None
            save_state(state, state_path)
        else:
            print("  Not all required platforms posted yet.")


if __name__ == "__main__":
    fix_state('state.json')
    fix_state('forwilma/state.json')
