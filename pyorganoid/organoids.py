import numpy as np
from .cells import *
from .modules import *
from .base import Organoid, Synapse
from .utils import generate_random_position


class SpikingNeuronOrganoid(Organoid):
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        """
    This class represents an organoid composed of spiking neurons. It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the spiking neuron modules.
    num_cells : int, optional
        The number of spiking neurons in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the spiking neurons. Defaults to '() => [0.5] * 10' if None.
    """
        super().__init__(environment)
        for i in range(num_cells):
            neuron = SpikingNeuronCell(position=generate_random_position(environment.dimensions, environment.size),
                                       input_data_func=input_data_func)
            neuron.add_module(SpikingNeuronModule(ml_model))
            self.add_agent(neuron)


class GrowthShrinkageOrganoid(Organoid):
    """
    This class represents an organoid composed of cells that grow and shrink based on machine learning predictions.
    It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the growth/shrinkage cell modules.
    num_cells : int, optional
        The number of growth/shrinkage cells in the organoid. Default is 10.
    initial_cell_volume : float, optional
        The initial volume of the cells. Default is 1.0. Use "None" for random volumes in the range [0.5, 1.5].
    growth_amount : float, optional
        The amount by which the cells grow at each time step. Default is 0.1.
    growth_variance : float, optional
        The variance of the growth amount. Default is 0.05.
    prediction_threshold : float, optional
        The threshold above which the cells grow and below which they shrink. Default is 0.5.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell volume if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, initial_cell_volume=1.0,
                 growth_amount=0.1, growth_variance=0.05, prediction_threshold=0.5, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            initial_volume = initial_cell_volume if initial_cell_volume is not None else np.random.uniform(0.5, 1.5)
            cell = GrowthShrinkageCell(position=generate_random_position(environment.dimensions, environment.size),
                                       initial_volume=initial_volume, input_data_func=input_data_func)
            cell.add_module(GrowthShrinkageModule(ml_model, growth_amount, growth_variance, prediction_threshold))
            self.add_agent(cell)


class DifferentiationOrganoid(Organoid):
    """
    This class represents an organoid composed of differentiating (e.g., neural network) cells.
    It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the differentiating cell modules.
    num_cells : int, optional
        The number of differentiating cells in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell state if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = DifferentiatingCell(position=generate_random_position(environment.dimensions, environment.size),
                                       input_data_func=input_data_func)
            cell.add_module(DifferentiationModule(ml_model))
            self.add_agent(cell)


class ChemotaxisOrganoid(Organoid):
    """
    This class represents an organoid composed of chemotactic cells. It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the chemotactic cell modules.
    num_cells : int, optional
        The number of chemotactic cells in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell gradient if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = ChemotacticCell(position=generate_random_position(environment.dimensions, environment.size),
                                   input_data_func=input_data_func)
            cell.add_module(ChemotaxisModule(ml_model))
            self.add_agent(cell)


class ImmuneResponseOrganoid(Organoid):
    """
    This class represents an organoid composed of immune cells that respond to external stimuli.
    It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the immune cell modules.
    num_cells : int, optional
        The number of immune cells in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell active state if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = ImmuneCell(position=generate_random_position(environment.dimensions, environment.size),
                              input_data_func=input_data_func)
            cell.add_module(ImmuneResponseModule(ml_model))
            self.add_agent(cell)


class SynapticPlasticityOrganoid(Organoid):
    """
    This class represents an organoid composed of neurons with synaptic plasticity (STDP).
    It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the synaptic plasticity cell modules.
    num_cells : int, optional
        The number of synaptic plasticity cells (i.e., neurons) in the organoid. Default is 10.
    num_synapses : int, optional
        The number of synapses connecting the plasticity cells. Default is 5.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning synapses if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, num_synapses=5, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = SynapticPlasticityCell(position=generate_random_position(environment.dimensions, environment.size),
                                          input_data_func=input_data_func)
            cell.add_module(SynapticPlasticityModule(ml_model, synapse=None))
            self.add_agent(cell)

        for _ in range(num_synapses):
            pre_cell, post_cell = np.random.choice(self.agents, 2, replace=False)
            synapse = Synapse(pre_neuron=pre_cell, post_neuron=post_cell)
            pre_cell.add_synapse(synapse)
            post_cell.add_module(SynapticPlasticityModule(ml_model, synapse=synapse))


class MetabolicOrganoid(Organoid):
    """
    This class represents an organoid composed of cells with metabolic modules.
    It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the metabolic cell modules.
    num_cells : int, optional
        The number of metabolic cells in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell energy if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = MetabolicCell(position=generate_random_position(environment.dimensions, environment.size),
                                 input_data_func=input_data_func)
            cell.add_module(MetabolicModule(ml_model))
            self.add_agent(cell)


class GeneRegulationOrganoid(Organoid):
    """
    This class represents an organoid composed of cells with gene regulation modules.
    It is a subclass of the base Organoid class.
    NOTE: In the future, this class could be extended to include sequence data and gene expression profiles.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides.
    ml_model : BaseMLModel
        The machine learning model used by the gene regulation cell modules.
    num_cells : int, optional
        The number of gene regulation cells in the organoid. Default is 10.
    regulation_variance : float, optional
        The maximum variance of the gene regulation. Default is 0.05.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning cell gene expression if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, regulation_variance=0.05, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = GeneRegulationCell(position=generate_random_position(environment.dimensions, environment.size),
                                      regulation_variance=regulation_variance, input_data_func=input_data_func)
            cell.add_module(GeneRegulationModule(ml_model))
            self.add_agent(cell)


class DigitClassificationOrganoid(Organoid):
    """
    This class represents an organoid composed of cells that classify handwritten digit images
    into class numbers (0-9). It is a subclass of the base Organoid class.

    Parameters
    ----------
    environment : Environment
        The environment in which the organoid resides. Typically a HandwritingEnvironment
        that streams handwritten digit images.
    ml_model : BaseMLModel
        The machine learning model used by the digit classification modules.
    num_cells : int, optional
        The number of digit classification cells in the organoid. Default is 10.
    input_data_func : callable, optional
        A function that generates input data for the cell. Defaults to returning the
        currently assigned digit image if None.
    """
    def __init__(self, environment, ml_model, num_cells=10, input_data_func=None):
        super().__init__(environment)
        for i in range(num_cells):
            cell = DigitClassificationCell(
                position=generate_random_position(environment.dimensions, environment.size),
                input_data_func=input_data_func
            )
            cell.environment = environment
            cell.add_module(DigitClassificationModule(ml_model))
            self.add_agent(cell)

    def classification_results(self, chronological=False):
        """
        Collect true and predicted class numbers recorded during the simulation.

        Parameters
        ----------
        chronological : bool, optional
            If True, interleave cell histories in scheduler order (all cells at step 0,
            then all cells at step 1, ...). If False, concatenate each cell's history
            in turn. Default is False.

        Returns
        -------
        tuple of np.ndarray
            (y_true, y_pred) arrays of class numbers.
        """
        if chronological and self.agents:
            n_steps = min(len(cell.predicted_history) for cell in self.agents)
            y_true = []
            y_pred = []
            for step in range(n_steps):
                for cell in self.agents:
                    y_true.append(cell.true_label_history[step])
                    y_pred.append(cell.predicted_history[step])
            return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)

        y_true = []
        y_pred = []
        for cell in self.agents:
            y_true.extend(cell.true_label_history)
            y_pred.extend(cell.predicted_history)
        return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)

    def classified_images(self, chronological=True):
        """
        Collect handwritten digit images with aligned true and predicted class numbers.

        Parameters
        ----------
        chronological : bool, optional
            If True, interleave cell histories in scheduler order. Default is True.

        Returns
        -------
        tuple of np.ndarray
            (images, y_true, y_pred)
        """
        y_true, y_pred = self.classification_results(chronological=chronological)
        if chronological and self.agents:
            n_steps = min(len(cell.digit_image_history) for cell in self.agents)
            images = []
            for step in range(n_steps):
                for cell in self.agents:
                    images.append(cell.digit_image_history[step])
            return np.asarray(images, dtype=float), y_true, y_pred

        images = []
        for cell in self.agents:
            images.extend(cell.digit_image_history)
        return np.asarray(images, dtype=float), y_true, y_pred

    def classification_accuracy(self):
        """
        Compute the fraction of handwritten digits classified to the correct class number.

        Returns
        -------
        float
            Classification accuracy in the range [0, 1].
        """
        y_true, y_pred = self.classification_results()
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    def plot_classification_results(self, filename=None, dpi=300, figsize=(8, 7)):
        """
        Plot a confusion matrix of true vs predicted handwritten digit classes.

        Parameters
        ----------
        filename : str, optional
            The filename to save the plot as (in PNG format). Default is None (view with plt.show()).
        dpi : int, optional
            The DPI (dots per inch) of the plot. Default is 300.
        figsize : tuple, optional
            The size of the figure (width, height) in inches. Default is (8, 7).

        Raises
        ------
        ImportError
            If Matplotlib is not installed.
        ValueError
            If the organoid has not classified any digits yet.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print("Error: Matplotlib is required to plot classification results.")
            raise e

        y_true, y_pred = self.classification_results()
        if len(y_true) == 0:
            raise ValueError("No classification results to plot. Run a simulation first.")

        classes = np.arange(10)
        matrix = np.zeros((10, 10), dtype=int)
        for true_label, predicted_label in zip(y_true, y_pred):
            if 0 <= true_label < 10 and 0 <= predicted_label < 10:
                matrix[true_label, predicted_label] += 1

        accuracy = self.classification_accuracy()
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(classes)
        ax.set_yticks(classes)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title(f"Handwritten Digit Classification\n{len(y_true)} samples, accuracy={accuracy:.1%}")
        for i in range(10):
            for j in range(10):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
            print(f'Classification results plot saved as "{filename}"')
            plt.close(fig)
        else:
            plt.show()
