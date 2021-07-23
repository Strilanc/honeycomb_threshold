import pathlib
from typing import Callable, List
import math
import subprocess
import tempfile

import networkx as nx
import numpy as np
import pymatching
import stim


USE_CORRELATIONS = False


def sample_decode_count_correct(*,
                                circuit: stim.Circuit,
                                num_shots: int,
                                use_internal_decoder: bool) -> int:
    """Counts how many times a decoder correctly predicts the logical frame of simulated runs."""

    num_dets = circuit.num_detectors
    num_obs = circuit.num_observables

    # Sample some runs with known solutions.
    det_obs_samples = circuit.compile_detector_sampler().sample(num_shots, append_observables=True)
    if num_obs == 0:
        det_samples = det_obs_samples[:, :]
        obs_samples = det_obs_samples[:, :0]
    else:
        det_samples = det_obs_samples[:, :-num_obs]
        obs_samples = det_obs_samples[:, -num_obs:]
    assert obs_samples.shape[0] == det_samples.shape[0]
    assert obs_samples.shape[1] == num_obs
    assert det_samples.shape[1] == num_dets

    # Have the decoder produce the solution from the symptoms.
    decode_method = decode_using_internal_decoder if use_internal_decoder else decode_using_pymatching
    predictions = decode_method(
        det_samples=det_samples,
        circuit=circuit,
    )

    # Count how many solutions were completely correct.
    assert predictions.shape == obs_samples.shape
    all_corrects = np.count_nonzero(predictions ^ obs_samples, axis=1) == 0
    return np.count_nonzero(all_corrects)


def decode_using_pymatching(circuit: stim.Circuit,
                            det_samples: np.ndarray,
                            ) -> np.ndarray:
    """Collect statistics on how often logical errors occur when correcting using detections."""
    error_model = circuit.detector_error_model()
    matching_graph = detector_error_model_to_pymatching_graph(error_model)

    num_shots = det_samples.shape[0]
    num_obs = circuit.num_observables
    num_dets = circuit.num_detectors
    assert det_samples.shape[1] == num_dets

    predictions = np.zeros(shape=(num_shots, num_obs), dtype=np.bool8)
    for k in range(num_shots):
        expanded_det = np.resize(det_samples[k], num_dets + 1)
        expanded_det[-1] = 0
        predictions[k] = matching_graph.decode(expanded_det)
    return predictions


def decode_using_internal_decoder(circuit: stim.Circuit,
                                  det_samples: np.ndarray,
                                  ) -> np.ndarray:
    num_shots = det_samples.shape[0]
    num_obs = circuit.num_observables
    assert det_samples.shape[1] == circuit.num_detectors
    error_model = circuit.detector_error_model(decompose_errors=True)

    with tempfile.TemporaryDirectory() as d:
        dem_file = f"{d}/model.dem"
        dets_file = f"{d}/shots.dets"
        out_file = f"{d}/out.predictions"

        with open(dem_file, "w") as f:
            print(error_model, file=f)
        with open(dets_file, "w") as f:
            for det_sample in det_samples:
                print("shot", file=f, end="")
                for k in np.nonzero(det_sample)[0]:
                    print(f" D{k}", file=f, end="")
                print(file=f)

        if not pathlib.Path("internal_decoder.binary").exists():
            raise RuntimeError(
                "You need an `internal_decoder.binary` file in the working directory to "
                "use `use_internal_decoder=True`.")

        command = (f"./internal_decoder.binary "
                   f"-mode fi_match_from_dem "
                   f"-dem_fname '{dem_file}' "
                   f"-dets_fname '{dets_file}' "
                   f"-ignore_distance_1_errors "
                   f"-ignore_undecomposed_errors "
                   f"-out '{out_file}'")
        if USE_CORRELATIONS:
            command += " -cheap_corr -edge_corr -node_corr"
        subprocess.check_output(command, shell=True)

        predictions = np.zeros(shape=(num_shots, num_obs), dtype=np.bool8)
        with open(out_file, "r") as f:
            for shot in range(num_shots):
                for obs_index in range(num_obs):
                    c = f.read(1)
                    assert c in '01'
                    predictions[shot, obs_index] = c == '1'
                assert f.read(1) == '\n'

        return predictions


def detector_error_model_to_pymatching_graph(model: stim.DetectorErrorModel) -> pymatching.Matching:
    """Convert stim error model into a pymatching graph."""
    det_offset = 0

    def _iter_model(m: stim.DetectorErrorModel, reps: int, callback: Callable[[float, List[int], List[int]], None]):
        nonlocal det_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _iter_model(instruction.body_copy(), instruction.repeat_count, callback)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets = []
                        frames = []
                        t: stim.DemTarget
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            else:
                                raise NotImplementedError()
                        p = instruction.args_copy()[0]
                        callback(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                    elif instruction.type == "detector":
                        pass
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

    g = nx.Graph()
    num_detectors = model.num_detectors
    for k in range(num_detectors):
        g.add_node(k)
    g.add_node(num_detectors, is_boundary=True)
    g.add_node(num_detectors + 1)
    for k in range(num_detectors + 1):
        g.add_edge(k, num_detectors + 1, weight=9999999999)

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 1:
            dets.append(num_detectors)
        if len(dets) != 2:
            return  # Just ignore correlated error mechanisms (e.g. Y errors / XX errors)
        g.add_edge(*dets, weight=-math.log(p), qubit_id=frame_changes)

    _iter_model(model, 1, handle_error)

    return pymatching.Matching(g)
