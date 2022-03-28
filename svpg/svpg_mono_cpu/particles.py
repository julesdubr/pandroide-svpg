from svpg.svpg_mono_cpu.agents import create_acquisition_agent

def create_particles(cfg, n_particles, env_agents):
    particles = list()
    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, prob_agent, critic_agent = create_acquisition_agent(cfg, env_agents[i], i)
        particles.append(
            {
                "acq_agent": acq_agent,
                "prob_agent": prob_agent,
                "critic_agent": critic_agent,
            }
        )

    return particles
