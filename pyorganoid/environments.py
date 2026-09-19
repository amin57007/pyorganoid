import numpy as np


class Environment:
    """
    Base class for environments. Environments can have conditions that change over time.
    Conditions are objects that have an update method.

    Parameters
    ----------
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int
        The number of dimensions of the environment.
    size : float
        The size of the environment.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    """
    def __init__(self, dimensions=3, size=100.0):
        self.conditions = []
        self.dimensions = dimensions
        self.size = size

    def update(self):
        """
        Update the environment.
        This method calls the update method of all conditions in the environment.
        """
        for condition in self.conditions:
            condition.update()

    def add_condition(self, condition):
        """
        Add a condition to the environment.

        Parameters
        ----------
        condition : object
            The condition to add to the environment.
        """
        self.conditions.append(condition)


class GradientEnvironment(Environment):
    """
    Environment with a gradient that can be used by agents to navigate.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    gradient_func : function
        The function that calculates the gradient at a given position.
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int, optional
        The number of dimensions of the environment.
    size : float, optional
        The size of the environment.
    gradient_func : function
        The function that calculates the gradient at a given position.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    get_gradient(position)
        Get the gradient at a given position.
    """
    def __init__(self, gradient_func, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.gradient_func = gradient_func

    def get_gradient(self, position):
        """
        Get the gradient at a given position.

        Parameters
        ----------
        position : array-like
            The position at which to calculate the gradient.

        Returns
        -------
        array-like
            The gradient at the given position.
        """
        return self.gradient_func(position)


class TemperatureEnvironment(Environment):
    """
    Environment with a temperature that can change over time.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    initial_temperature : float
        The initial temperature of the environment.
    temperature_range : tuple, optional
        The range of temperatures that the environment can have. Default is None.
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.
    temperature : float
        The temperature of the environment.
    temperature_range : tuple
        The range of temperatures that the environment can have (optional).

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    get_temperature()
        Get the temperature of the environment.
    set_temperature(temperature)
        Set the temperature of the environment.
    """
    def __init__(self, initial_temperature, temperature_range=None, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.temperature = initial_temperature
        self.temperature_range = temperature_range

    def update(self):
        """
        Update the environment.
        This method updates the temperature of the environment if a temperature range is specified.
        """
        if self.temperature_range is not None:
            self.temperature = np.random.uniform(*self.temperature_range)
        super().update()

    def get_temperature(self):
        """
        Get the temperature of the environment.

        Returns
        -------
        float
            The temperature of the environment.
        """
        return self.temperature

    def set_temperature(self, temperature):
        """
        Set the temperature of the environment.

        Parameters
        ----------
        temperature : float
            The temperature to set
        """
        self.temperature = temperature


class StochasticEnvironment(Environment):
    """
    Environment with a stochastic noise level that can change over time.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    noise_level : float
        The noise level of the environment
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int, optional
        The number of dimensions of the environment.
    size : float, optional
        The size of the environment.
    noise_level : float
        The noise level of the environment.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    get_noise()
        Get the noise level of the environment.
    """
    def __init__(self, noise_level, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.noise_level = noise_level

    def get_noise(self):
        """
        Get the noise level of the environment.

        Returns
        -------
        float
            The noise level of the environment.
        """
        return np.random.normal(0, self.noise_level)


class ChemicalGradientEnvironment(Environment):
    """
    Environment with a chemical gradient that can be used by agents to navigate.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    gradient_func : function
        The function that calculates the gradient at a given position. Default is np.linalg.norm.
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int, optional
        The number of dimensions of the environment.
    size : float, optional
        The size of the environment.
    gradient_func : function
        The function that calculates the gradient at a given position.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    get_concentration(position)
        Get the concentration of the chemical at a given position.
    """
    def __init__(self, gradient_func=np.linalg.norm, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.gradient_func = gradient_func

    def get_concentration(self, position):
        """
        Get the concentration of the chemical at a given position.

        Parameters
        ----------
        position : array-like
            The position at which to calculate the concentration.
        """
        return self.gradient_func(position)


class ElectricFieldEnvironment(Environment):
    """
    Environment with an electric field that can be used by agents to navigate.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    field_strength : float
        The strength of the electric field.
    dimensions : int, optional
        The number of dimensions of the environment. Default is 3.
    size : float, optional
        The size of the environment. Default is 100.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int, optional
        The number of dimensions of the environment.
    size : float, optional
        The size of the environment.
    field_strength : float
        The strength of the electric field.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    get_field_effect(position)
        Get the effect of the electric field at a given position.
    """
    def __init__(self, field_strength, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.field_strength = field_strength

    def get_field_effect(self, position):
        """
        Get the effect of the electric field at a given position.

        Parameters
        ----------
        position : array-like
            The position at which to calculate the field effect.
        """
        return self.field_strength * position  # May be overridden in subclasses for more complex behavior


class HandwritingEnvironment(Environment):
    """
    Environment that presents a stream of handwritten digit images (e.g., 0-9 class numbers).
    Cells can pull the next image/label pair during each simulation step.
    It is a subclass of the base Environment class.

    Parameters
    ----------
    images : array-like
        Handwritten digit images with shape (n_samples, n_features) or (n_samples, height, width).
    labels : array-like
        True class numbers for each image, with shape (n_samples,).
    dimensions : int, optional
        The number of dimensions of the environment. Default is 2.
    size : float, optional
        The size of the environment. Default is 50.0.

    Attributes
    ----------
    conditions : list
        List of conditions in the environment.
    dimensions : int
        The number of dimensions of the environment.
    size : float
        The size of the environment.
    images : np.ndarray
        Flattened handwritten digit images.
    labels : np.ndarray
        True class numbers for each image.
    index : int
        Index of the next sample to present.
    current_image : np.ndarray
        The most recently presented digit image.
    current_label : int
        The most recently presented true class number.

    Methods
    -------
    update()
        Update the environment.
    add_condition(condition)
        Add a condition to the environment
    next_sample()
        Return the next handwritten digit image and its true class number.
    reset()
        Reset the sample stream to the first digit.
    """
    def __init__(self, images, labels, dimensions=2, size=50.0):
        super().__init__(dimensions, size)
        self.images = np.asarray(images, dtype=float)
        if self.images.ndim > 2:
            self.images = self.images.reshape(len(self.images), -1)
        self.labels = np.asarray(labels).reshape(-1)
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must contain the same number of samples.")
        if len(self.images) == 0:
            raise ValueError("HandwritingEnvironment requires at least one handwritten digit sample.")
        self.index = 0
        self.current_image = self.images[0]
        self.current_label = int(self.labels[0])

    def next_sample(self):
        """
        Return the next handwritten digit image and its true class number.
        The stream wraps around after the last sample.

        Returns
        -------
        tuple
            (image, label) where image is a 1D array of pixels and label is the true class number.
        """
        image = self.images[self.index]
        label = int(self.labels[self.index])
        self.current_image = image
        self.current_label = label
        self.index = (self.index + 1) % len(self.images)
        return image, label

    def reset(self):
        """
        Reset the sample stream to the first digit.
        """
        self.index = 0
        self.current_image = self.images[0]
        self.current_label = int(self.labels[0])


class StimParam:
    """
    Stimulation parameter block modeled on FinalSpark Neuroplatform StimParam.

    Parameters
    ----------
    index : int, optional
        Electrode index in the 128-channel MEA map. Default is 0.
    trigger_key : int, optional
        Trigger key in [0, 15]. Default is 0.
    enable : bool, optional
        Whether this parameter is armed. Default is True.
    phase_amplitude1 : float, optional
        First-phase amplitude (abstract uA). Default is 1.0.
    phase_duration1 : float, optional
        First-phase duration in microseconds. Default is 100.0.
    phase_amplitude2 : float, optional
        Second-phase amplitude (abstract uA). Default is 1.0.
    phase_duration2 : float, optional
        Second-phase duration in microseconds. Default is 100.0.
    nb_pulse : int, optional
        Number of pulses in the train. Default is 1.
    stim_shape : str, optional
        Pulse shape name. Default is "Biphasic".
    """
    def __init__(self, index=0, trigger_key=0, enable=True, phase_amplitude1=1.0,
                 phase_duration1=100.0, phase_amplitude2=1.0, phase_duration2=100.0,
                 nb_pulse=1, stim_shape="Biphasic"):
        if not 0 <= int(index) <= 127:
            raise ValueError("electrode index must be in [0, 127].")
        if not 0 <= int(trigger_key) <= 15:
            raise ValueError("trigger_key must be in [0, 15].")
        self.index = int(index)
        self.trigger_key = int(trigger_key)
        self.enable = bool(enable)
        self.phase_amplitude1 = float(phase_amplitude1)
        self.phase_duration1 = float(phase_duration1)
        self.phase_amplitude2 = float(phase_amplitude2)
        self.phase_duration2 = float(phase_duration2)
        self.nb_pulse = int(nb_pulse)
        self.stim_shape = stim_shape


class NeuroplatformEnvironment(Environment):
    """
    Simulated FinalSpark Neuroplatform MEA.

    Hardware analog: 4 MEAs x 4 organoids x 8 electrodes = 128 channels, 16 triggers,
    biphasic current stimulation, and a closed-loop spike-count readout after a 200 ms window.
    This class does not connect to living tissue; it provides the same control/readout shape
    so a decoder trained here can later be pointed at ``intan.read_count()``.

    Parameters
    ----------
    organoid_electrodes : array-like, optional
        Channel indices used by the active organoid. Default is electrodes 0-7 (MEA 0, site 0).
    window_steps : int, optional
        Inner integrate-and-fire steps that stand in for the ~200 ms count window. Default is 10.
    noise_std : float, optional
        Gaussian current noise added on each inner step. Default is 0.02.
    window_ms : float, optional
        Documented closed-loop window in milliseconds. Default is 200.0.
    latency_ms : float, optional
        Documented Python-to-hardware latency in milliseconds. Default is 40.0.
    dimensions : int, optional
        Spatial dimensions of the simulated dish. Default is 2.
    size : float, optional
        Spatial extent of the dish. Default is 50.0.
    """
    N_CHANNELS = 128
    N_TRIGGERS = 16
    ELECTRODES_PER_ORGANOID = 8
    N_MEAS = 4
    ORGANOIDS_PER_MEA = 4

    def __init__(self, organoid_electrodes=None, window_steps=10, noise_std=0.02,
                 window_ms=200.0, latency_ms=40.0, dimensions=2, size=50.0):
        super().__init__(dimensions, size)
        if organoid_electrodes is None:
            organoid_electrodes = list(range(self.ELECTRODES_PER_ORGANOID))
        self.organoid_electrodes = np.asarray(organoid_electrodes, dtype=int)
        if np.any((self.organoid_electrodes < 0) | (self.organoid_electrodes >= self.N_CHANNELS)):
            raise ValueError("organoid_electrodes must be in [0, 127].")
        self.window_steps = int(window_steps)
        self.noise_std = float(noise_std)
        self.window_ms = float(window_ms)
        self.latency_ms = float(latency_ms)
        self.pattern_trigger = 0
        self.stim_params = []
        self.count_triggers = [0]
        self.stim_amplitudes = np.zeros(self.N_CHANNELS, dtype=float)
        self.spike_counts = np.zeros(self.N_CHANNELS, dtype=int)
        self.last_trigger = np.zeros(self.N_TRIGGERS, dtype=np.uint8)
        self.last_encoded_amplitudes = np.zeros(len(self.organoid_electrodes), dtype=float)

    @staticmethod
    def site_channels(mea=0, organoid=0):
        """
        Return the 8 electrode indices for one organoid site.

        Parameters
        ----------
        mea : int, optional
            MEA index in [0, 3]. Default is 0.
        organoid : int, optional
            Organoid index on that MEA in [0, 3]. Default is 0.

        Returns
        -------
        list of int
            Eight channel indices.
        """
        if not 0 <= mea < NeuroplatformEnvironment.N_MEAS:
            raise ValueError("mea must be in [0, 3].")
        if not 0 <= organoid < NeuroplatformEnvironment.ORGANOIDS_PER_MEA:
            raise ValueError("organoid must be in [0, 3].")
        start = mea * 32 + organoid * NeuroplatformEnvironment.ELECTRODES_PER_ORGANOID
        return list(range(start, start + NeuroplatformEnvironment.ELECTRODES_PER_ORGANOID))

    @staticmethod
    def encode_image(image, n_electrodes=8):
        """
        Encode a handwritten digit into one amplitude per electrode.

        8x8 (or 64-d) images use row means so each of 8 electrodes receives one row.
        Other lengths are resampled to n_electrodes bins.

        Parameters
        ----------
        image : array-like
            Digit pixels, typically shape (8, 8) or (64,).
        n_electrodes : int, optional
            Number of stimulation channels. Default is 8.

        Returns
        -------
        np.ndarray
            Amplitude vector of length n_electrodes in [0, 1].
        """
        pixels = np.asarray(image, dtype=float).flatten()
        if pixels.size == 0:
            raise ValueError("image must contain pixels.")
        if pixels.max() > 1.0:
            pixels = pixels / 16.0 if pixels.max() <= 16.0 else pixels / pixels.max()
        pixels = np.clip(pixels, 0.0, 1.0)
        if pixels.size == 64 and n_electrodes == 8:
            return np.asarray(pixels.reshape(8, 8).mean(axis=1), dtype=float)
        bins = np.array_split(pixels, n_electrodes)
        return np.asarray([float(np.mean(b)) if len(b) else 0.0 for b in bins], dtype=float)

    def send_stimparam(self, params):
        """
        Upload stimulation parameters (simulated analog of Intan send_stimparam).

        Parameters
        ----------
        params : list of StimParam
            Parameter blocks to arm.
        """
        self.stim_params = list(params)

    def set_count(self, trigger_ids):
        """
        Choose which trigger keys start a new spike-count window.

        Parameters
        ----------
        trigger_ids : list of int
            Trigger keys in [0, 15].
        """
        self.count_triggers = [int(t) for t in trigger_ids]

    def arm_stim(self, amplitudes, trigger_key=None):
        """
        Build StimParam blocks that map a pattern onto the organoid electrodes.

        Parameters
        ----------
        amplitudes : array-like
            One amplitude per organoid electrode.
        trigger_key : int, optional
            Trigger used to fire the whole pattern. Default is pattern_trigger (0).
        """
        amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
        if len(amplitudes) != len(self.organoid_electrodes):
            raise ValueError("amplitudes length must match organoid_electrodes.")
        if trigger_key is None:
            trigger_key = self.pattern_trigger
        self.last_encoded_amplitudes = amplitudes.copy()
        params = []
        for electrode, amplitude in zip(self.organoid_electrodes, amplitudes):
            params.append(StimParam(index=int(electrode), trigger_key=int(trigger_key),
                                    enable=True, phase_amplitude1=float(amplitude),
                                    phase_amplitude2=float(amplitude), nb_pulse=1))
        self.send_stimparam(params)
        self.set_count([int(trigger_key)])

    def send(self, trigger_mask):
        """
        Fire triggers. Enabled StimParams bound to those keys inject current.

        Parameters
        ----------
        trigger_mask : array-like
            Length-16 uint8 vector; 1 means that trigger key is sent.

        Returns
        -------
        np.ndarray
            The 16-d trigger mask that was applied.
        """
        mask = np.asarray(trigger_mask, dtype=np.uint8).reshape(-1)
        if mask.size != self.N_TRIGGERS:
            raise ValueError("trigger mask must have 16 entries (keys 0-15).")
        fired = {i for i, flag in enumerate(mask) if flag}
        self.last_trigger = mask.copy()
        if fired.intersection(self.count_triggers):
            self.spike_counts[:] = 0
        self.stim_amplitudes[:] = 0.0
        for param in self.stim_params:
            if param.enable and param.trigger_key in fired:
                n_pulse = max(int(param.nb_pulse), 1)
                self.stim_amplitudes[param.index] = abs(param.phase_amplitude1) * n_pulse
        return mask

    def read_count(self):
        """
        Return spike counts on all 128 channels since the last counting trigger.

        Returns
        -------
        np.ndarray
            Integer vector of length 128.
        """
        return self.spike_counts.copy()

    def record_spike(self, electrode_index):
        """
        Increment the spike counter for one electrode.

        Parameters
        ----------
        electrode_index : int
            Channel index in [0, 127].
        """
        self.spike_counts[int(electrode_index)] += 1

    def organoid_counts(self, counts=None):
        """
        Slice the 128-d count vector down to this organoid's electrodes.

        Parameters
        ----------
        counts : array-like, optional
            Full 128-d vector. Default is the current buffer.

        Returns
        -------
        np.ndarray
            Count vector for the active organoid electrodes.
        """
        if counts is None:
            counts = self.spike_counts
        return np.asarray(counts, dtype=float)[self.organoid_electrodes]
