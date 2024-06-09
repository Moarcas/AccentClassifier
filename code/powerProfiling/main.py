import torch
import numpy as np
import pytest
from snntorch.energy_estimation.estimate_energy import estimate_energy
from snntorch.energy_estimation.device_profile_registry import DeviceProfileRegistry
from snntorch.energy_estimation.layer_parameter_event_calculator import (LayerParameterEventCalculator,
                                                                         synapse_neuron_count_for_linear,
                                                                         count_events_for_linear)


from dataUtils import get_batch
from networks import CNN_mfcc, CNN_amplitude, SNN_mfcc


def get_estimation_CNN_mfcc():
    model = CNN_mfcc()
    batch = get_batch(feature_type='mfcc', batch_size=64)

    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("cpu-test2", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     True)

    # get the layer summaries
    summary_list = estimate_energy(model=model, input_data=batch,
                                   devices="cpu-test2",
                                   include_bias_term_in_events=False)

    print(summary_list)


def get_estimation_CNN_amplitude():
    model = CNN_amplitude()
    batch = get_batch(feature_type='amplitude', batch_size=64)

    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("cpu-test2", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     False)

    # get the layer summaries
    summary_list = estimate_energy(model=model, input_data=batch,
                                   devices="cpu-test2",
                                   include_bias_term_in_events=False)

    print(summary_list)


def get_estimation_SNN_mfcc():
    model = SNN_mfcc()
    batch = get_batch(feature_type='mfcc', batch_size=128)

    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("cpu-test2", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     False)

    # get the layer summaries
    summary_list = estimate_energy(model=model, input_data=batch,
                                   devices="cpu-test2",
                                   include_bias_term_in_events=False)

    print(summary_list)


# get_estimation_CNN_mfcc()

# get_estimation_CNN_amplitude()

get_estimation_SNN_mfcc()
