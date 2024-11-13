import json
import matplotlib.pyplot as plt
with open('stats.jsonl', 'r') as f:
    stats = f.readlines()

achievements = {}
for episode in stats[-20:]:
    episode = json.loads(episode)
    for key, value in episode.items():
        if not key.startswith('achievement'):
            continue

        if value == 0:
            continue
        if '_'.join(key.split('_')[1:]) not in achievements:
            achievements['_'.join(key.split('_')[1:])] = 0

        achievements['_'.join(key.split('_')[1:])] += value


keys = list(achievements.keys())
values = list(achievements.values())

plt.bar(keys, values)
plt.xticks(rotation=90)
plt.title('Achievements for last evaluation')
plt.savefig("achievements.png", bbox_inches='tight')
plt.show()