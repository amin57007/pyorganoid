import numpy as np


class Scheduler:
    """
    Base class for simulation schedulers. Can be used to simulate the behavior of an organoid in a given environment.

    Parameters
    ----------
    organoid : Organoid
        The organoid to simulate.

    Attributes
    ----------
    organoid : Organoid
        The organoid to simulate.

    Methods
    -------
    simulate(steps)
        Simulate the organoid's behavior over a number of steps.
    """
    def __init__(self, organoid):
        self.organoid = organoid

    def simulate(self, steps):
        """
        Simulate the organoid's behavior over a number of steps.

        Parameters
        ----------
        steps : int
            The number of steps to run the simulation.
        """
        for step in range(steps):
            print(f"Step {step+1}/{steps}")
            self.organoid.environment.update()
            for agent in self.organoid.agents:
                agent.update()


class StochasticScheduler(Scheduler):
    """
    Scheduler that simulates the organoid's behavior with a stochastic update rule.
    The update rule is applied to each agent with a 50% chance.
    It is a subclass of the base Scheduler class.

    Parameters
    ----------
    organoid : Organoid
        The organoid to simulate.

    Attributes
    ----------
    organoid : Organoid
        The organoid to simulate.

    Methods
    -------
    simulate(steps)
        Simulate the organoid's behavior over a number of steps with a stochastic update rule.
    """
    def simulate(self, steps):
        """
        Simulate the organoid's behavior over a number of steps with a stochastic update rule.
        The update rule is applied to each agent with a 50% chance.

        Parameters
        ----------
        steps : int
            The number of steps to run the simulation.
        """
        for step in range(steps):
            print(f"Step {step+1}/{steps}")
            self.organoid.environment.update()
            for agent in self.organoid.agents:
                if np.random.rand() > 0.5:  # 50% chance to update the agent
                    agent.update()


class PriorityScheduler(Scheduler):
    """
    Scheduler that simulates the organoid's behavior based on a priority system.
    Agents are updated in order of decreasing priority.
    It is a subclass of the base Scheduler class.

    Parameters
    ----------
    organoid : Organoid
        The organoid to simulate.
    priorities : dict
        A dictionary of agent priorities.

    Attributes
    ----------
    organoid : Organoid
        The organoid to simulate.
    priorities : dict
        A dictionary of agent priorities.

    Methods
    -------
    simulate(steps)
        Simulate the organoid's behavior over a number of steps based on a priority system.
    """
    def __init__(self, organoid, priorities):
        super().__init__(organoid)
        self.priorities = priorities

    def simulate(self, steps):
        """
        Simulate the organoid's behavior over a number of steps based on a priority system.
        Agents are updated in order of decreasing priority.

        Parameters
        ----------
        steps : int
            The number of steps to run the simulation.
        """
        for step in range(steps):
            print(f"Step {step+1}/{steps}")
            self.organoid.environment.update()
            sorted_agents = sorted(self.organoid.agents, key=lambda agent: self.priorities.get(agent, 0), reverse=True)
            for agent in sorted_agents:
                agent.update()


class ParallelScheduler(Scheduler):
    """
    Scheduler that runs the simulation in parallel using multiple CPU cores.
    The class has not yet been implemented.
    It is a subclass of the base Scheduler class.

    Parameters
    ----------
    organoid : Organoid
        The organoid to simulate.

    Attributes
    ----------
    organoid : Organoid
        The organoid to simulate.

    Methods
    -------
    simulate(steps)
        Simulate the organoid's behavior over a number of steps.
    """

    def simulate(self, steps):
        """
        Simulate the organoid's behavior over a number of steps.

        This method is intended to run the simulation in parallel using multiple
        CPU cores. Currently, this function is not yet implemented. Using joblib may
        be the most feasible way to implement parallel simulation over multiprocessing;
        however, the cell history is neither preserved nor updated correctly when executed in parallel.

        Parameters
        ----------
        steps : int
            The number of steps to run the simulation.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Parallel simulation is not yet implemented. Please use a different scheduler.")


class NeuroplatformScheduler(Scheduler):
    """
    Closed-loop scheduler modeled on FinalSpark trigger -> wait -> read_count.

    Each trial: encode image, upload StimParams, send a 16-d trigger, run window_steps
    of integrate-and-fire updates, then return the 128-d spike-count vector.

    Parameters
    ----------
    organoid : NeuroplatformOrganoid
        Organoid whose environment is a NeuroplatformEnvironment.
    """
    def present(self, image):
        """
        Present one stimulus and return the full 128-channel spike-count vector.

        Parameters
        ----------
        image : array-like
            Handwritten digit pixels.

        Returns
        -------
        np.ndarray
            Integer vector of length 128.
        """
        environment = self.organoid.environment
        amplitudes = environment.encode_image(image, n_electrodes=len(environment.organoid_electrodes))
        environment.arm_stim(amplitudes)
        trigger = np.zeros(environment.N_TRIGGERS, dtype=np.uint8)
        trigger[environment.pattern_trigger] = 1
        environment.send(trigger)
        for _ in range(environment.window_steps):
            for agent in self.organoid.agents:
                agent.update()
        return environment.read_count()

    def simulate(self, images, decoder=None, verbose=True):
        """
        Present a sequence of images and optionally decode each spike-count vector.

        Parameters
        ----------
        images : array-like
            Collection of digit images.
        decoder : object, optional
            Object with a ``predict`` method on the 8-d organoid count vector.
        verbose : bool, optional
            Print trial progress. Default is True.

        Returns
        -------
        dict
            counts (n, 128), organoid_counts (n, 8), and predictions if a decoder is given.
        """
        images = list(images)
        all_counts = []
        all_organoid = []
        predictions = []
        n = len(images)
        for i, image in enumerate(images):
            if verbose and (i == 0 or (i + 1) % 100 == 0 or i + 1 == n):
                print(f"Trial {i + 1}/{n}")
            counts = self.present(image)
            organoid_counts = self.organoid.environment.organoid_counts(counts)
            all_counts.append(counts)
            all_organoid.append(organoid_counts)
            if decoder is not None:
                predictions.append(int(np.asarray(decoder.predict(organoid_counts.reshape(1, -1))).reshape(-1)[0]))
        result = {
            "counts": np.asarray(all_counts, dtype=int),
            "organoid_counts": np.asarray(all_organoid, dtype=float),
        }
        if decoder is not None:
            result["predictions"] = np.asarray(predictions, dtype=int)
        return result
