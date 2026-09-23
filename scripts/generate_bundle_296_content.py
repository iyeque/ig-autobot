#!/usr/bin/env python3
"""Generate bundle 296 content (reflection + captions) without AI Horde.
Writes master_reflection and platform captions directly to state.json."""
import json
from pathlib import Path

state_path = Path('state.json')
state = json.load(state_path.open('r', encoding='utf-8'))

active = state.get('active_bundle', {})
topic = active.get('topic', 'The Feedback Loop We Live In #18')

# Master reflection for bundle 296 (systems_psychology pillar)
master_reflection = """We live in systems designed to capture our attention and return us to the same loops.

The feedback loop is elegant in its brutality: a trigger, a behavior, a reward, and the craving that pulls us back. It is the same mechanism that keeps a teenager refreshing Instagram and a day trader checking portfolio prices every thirty seconds.

What makes us human is not our vulnerability to these loops—that is simply our neurobiology. What makes us human is our capacity to observe them, name them, and redesign the system around them.

The circuit and the tree root are the same image: one is technology designed to loop endlessly, the other is life that grows through recursion—each branch splitting, seeking light, returning to the same pattern of reaching.

Productive failure is not failing productively. It is recognizing that the loop exists, stepping outside it long enough to ask whether the reward you are chasing is actually yours.

The most radical act of systems psychology is not optimization. It is refusal.

#DigitalWellness #ScreenTime #TheNineStitches #SystemsPsychology"""

# Platform-specific captions
captions = {
    "instagram": f"""{topic}

We are caught in loops we didn't design. The question isn't whether the loop works—it's working on us.

Save this if you're ready to redesign your digital environment.

#DigitalWellness #ScreenTime #MentalHealth #TheNineStitches #Productivity #Focus""",

    "linkedin": f"""{topic}

The most overlooked skill in knowledge work is not deep work—it's loop awareness.

Every platform you use is engineered to return you to the same trigger-reward cycle. Understanding the architecture of your attention is the first step toward reclaiming it.

The circuit diagram and the tree root look alike because both are recursive systems. One drains energy. The other generates it.

Which loop are you optimizing for?

#DigitalWellness #KnowledgeWork #AttentionEconomy #Leadership""",

    "bluesky": f"""The loop isn't broken. It's working exactly as designed—on us.

Awareness is the off-ramp. 🔄

#DigitalWellness #ScreenTime""",

    "threads": f"""We talk about digital wellness like it's a personal discipline problem.

It's not. It's a systems design problem.

The loop works. The question is: who is it working for?""",

    "pinterest": f"""The Feedback Loop We Live In

Understanding how digital systems capture attention is the first step to designing better boundaries.

#DigitalWellness #ScreenTime #ProductivityTips #MindfulTech""",

    "youtube": f"""We live in systems designed to capture our attention and return us to the same loops.

In this short, I break down the psychology of digital feedback loops—and why awareness is the only off-ramp that actually works.

Like and subscribe for more on digital wellness and attention architecture.

#DigitalWellness #ScreenTime #MentalHealth #Shorts""",
}

# Update state
active['master_reflection'] = master_reflection
active['captions'] = captions

state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
print('✅ Generated content for bundle 296')
print(f'Master reflection: {len(master_reflection)} chars')
for p, c in captions.items():
    print(f'  {p}: {len(c)} chars')
