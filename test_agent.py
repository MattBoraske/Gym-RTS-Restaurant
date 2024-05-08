#########
# TO DO #
#########

# complete a single run (one day of the restaurant) with the trained agent
    # will likely require modifying env.render() to print out more information

'''
BELOW IS WHAT WAS USED FOR THE DEMO GOLEFTENV

# Test the trained agent
# using the vecenv
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
'''