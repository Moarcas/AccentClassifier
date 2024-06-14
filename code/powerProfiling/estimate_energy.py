import sys
from snntorch.energy_estimation.estimate_energy import estimate_energy
from snntorch.energy_estimation.device_profile_registry import DeviceProfileRegistry
from dataUtils import get_batch
from networks import get_CNN_mfcc, get_CNN_amplitude, get_SNN_mfcc


def get_estimation_CNN_mfcc():
    model = get_CNN_mfcc()
    batch = get_batch(feature_type='mfcc', batch_size=64, arhitecture='cnn')

    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("vn", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     True)

    # get the layer summaries
    summary_list = estimate_energy(model=model, input_data=batch,
                                   devices="vn",
                                   network_requires_last_dim_as_time=False,
                                   include_bias_term_in_events=True)

    print(summary_list)


def get_estimation_CNN_amplitude():
    model = get_CNN_amplitude()
    batch = get_batch(feature_type='amplitude', batch_size=64, arhitecture='cnn')

    energy_per_synapse_event = 8.6e-9
    energy_per_neuron_event = 8.6e-9
    DeviceProfileRegistry.add_device("vn", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     False)

    # get the layer summaries
    summary_list = estimate_energy(model=model, input_data=batch,
                                   devices="vn",
                                   network_requires_last_dim_as_time=False,
                                   include_bias_term_in_events=True)

    print(summary_list)


def get_estimation_SNN_mfcc():
    model = get_SNN_mfcc()
    batch = get_batch(feature_type='mfcc', batch_size=64, arhitecture='snn')

    energy_per_synapse_event = 1e-9
    energy_per_neuron_event = 1e-9
    DeviceProfileRegistry.add_device("neuromorphic", energy_per_synapse_event,
                                     energy_per_neuron_event,
                                     True)

    # get the layer summaries
    summary_list_neuromorphic = estimate_energy(model=model,
                                                input_data=batch,
                                                devices="neuromorphic",
                                                network_requires_last_dim_as_time=True,
                                                include_bias_term_in_events=True)
    print('Neuromorphic arhitecture', summary_list_neuromorphic)


def main():
    available_models = ['CNN_mfcc', 'CNN_amplitude', 'SNN_mfcc']
    if len(sys.argv) != 2 or sys.argv[1] not in available_models:
        print("Usage: python3 convert_to_spikes.py <feature> <encoding>")
        print("Please provide one argument when running the script.")
        print("Available arguments for models:", ", ".join(available_models))
        return

    model_type = sys.argv[1]

    if model_type == 'CNN_mfcc':
        get_estimation_CNN_mfcc()
    elif model_type == 'CNN_amplitude':
        get_estimation_CNN_amplitude()
    else:
        get_estimation_SNN_mfcc()


if __name__ == '__main__':
    main()
