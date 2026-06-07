import subprocess

baselines = ["none", "avg", "value", "gae"]
envs = ["CartPole-v1", "LunarLander-v3"]
seeds = [0, 892735897, 142312]

# Train all baselines with multiple seeds
for env in envs:
    for baseline in baselines:
        for seed in seeds:
            cmd = [
                "python",
                "rl_exercises/week_6/actor_critic.py",
                f"--config-name=actor_critic_{baseline}",
                f"env.name={env}",
                f"seed={seed}",
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
