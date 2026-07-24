import sys, json, os
sys.path.insert(0, '.')
from shared_utils import save_state, load_state

state = load_state('state.json')

bundle_3001 = {
    'post_id': 3001,
    'timestamp': '20260722_014557',
    'image': 'images/post_20260602_084216.jpg',
    'reel': 'reels/reel_3001_unique.mp4',
    'story': 'images/story.jpg' if os.path.exists('images/story.jpg') else None,
    'carousel': [],
    'pillar': 'mindset',
    'topic': 'Curated perfection has starved us',
    'format': 'reel',
    'platforms_posted': [],
    'trailer_for': 'Curated perfection has starved us',
    'hook_frame': 'Curated perfection has starved us.\nStop waiting for permission.',
    'captions': {
        'instagram': (
            "Raw proof: notes scattered across my desk at midnight because healing isn't linear and neither is this essay I keep rewriting.\n\n"
            "The myth of overnight transformation sells merch but not peace. Real work happens between the viral moments—unseen drafts, deleted paragraphs, conversations with myself about whether any of it matters.\n\n"
            "Stop glorifying brokenness as your brand identity when you could be building actual capacity to hold complexity without collapsing into poetry or performance.\n\n"
            "This week's practice: One thing you're avoiding doing \"correctly.\" Do it anyway. Then do it again tomorrow. Let me know what surfaces when you stop waiting for permission to exist imperfectly.\n\n"
            "Save and share this if it resonates.\n\n"
            "#TheNineStitches #QuoteCollectors #BookQuotes #ThoughtfulLiving"
        ),
        'threads': (
            "Raw proof: notes scattered across my desk at midnight because healing isn't linear.\n\n"
            "The myth of overnight transformation sells merch but not peace. Real work happens between the viral moments—unseen drafts, deleted paragraphs, conversations with myself about whether any of it matters.\n\n"
            "Stop glorifying brokenness as your brand identity when you could be building actual capacity to hold complexity without collapsing into performance.\n\n"
            "This week's practice: One thing you're avoiding doing \"correctly.\" Do it anyway.\n\n"
            "Want to read more?... check out my LinkedIn"
        ),
        'bluesky': (
            "Raw proof: notes scattered across my desk at midnight because healing isn't linear.\n\n"
            "The myth of overnight transformation sells merch but not peace. Real work happens between the viral moments—unseen drafts, deleted paragraphs.\n\n"
            "Stop glorifying brokenness when you could be building actual capacity.\n\n"
            "Want to read more?... check out my LinkedIn"
        ),
        'linkedin': (
            "Curated perfection has starved us.\n\n"
            "Raw proof: notes scattered across my desk at midnight because healing isn't linear and neither is any strategy worth pursuing.\n\n"
            "The myth of overnight transformation sells merch but not peace. Real work happens between the visible milestones—unseen drafts, deleted paragraphs, conversations with yourself about whether any of it matters.\n\n"
            "Stop glorifying brokenness as your brand identity when you could be building actual capacity to hold complexity without collapsing into performance.\n\n"
            "This week's practice: Identify one thing you're avoiding doing \"correctly.\" Do it anyway. Then do it again tomorrow.\n\n"
            "What surfaces when you stop waiting for permission to exist imperfectly?\n\n"
            "#TheNineStitches #BehaviorPatterns #GrowthMindset"
        ),
        'youtube': (
            "Curated perfection has starved us. The myth of overnight transformation sells merch but not peace. Real work happens between the viral moments—unseen drafts, deleted paragraphs, and quiet persistence.\n\n"
            "Stop glorifying brokenness when you could be building actual capacity to hold complexity.\n\n"
            "#TheNineStitches #MindsetScience #BehaviorPatterns"
        ),
        'pinterest': (
            "Curated perfection has starved us. Raw proof: notes scattered across my desk at midnight because healing isn't linear. The myth of overnight transformation sells merch but not peace. Real work happens between the viral moments.\n\n"
            "Stop glorifying brokenness as your brand identity when you could be building actual capacity to hold complexity without collapsing into performance.\n\n"
            "#MentalHealth #GrowthMindset #Authenticity #TheNineStitches #DailyWisdom"
        )
    }
}

queue = [q for q in state.get('content_queue', []) if isinstance(q, dict) and q.get('post_id') != 3001]
state['active_bundle'] = bundle_3001
state['content_queue'] = queue

save_state(state, 'state.json')
print('Active bundle updated to 3001 successfully!')
