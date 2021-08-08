import pathlib
import sys
from typing import Callable, List, Optional
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
    all_corrects = np.all(predictions == obs_samples, axis=1)
    return np.count_nonzero(all_corrects)


def decode_using_pymatching(circuit: stim.Circuit,
                            det_samples: np.ndarray,
                            ) -> np.ndarray:
    """Collect statistics on how often logical errors occur when correcting using detections."""
    error_model = circuit.detector_error_model(decompose_errors=True)
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


def internal_decoder_path() -> Optional[str]:
    for possible_dirs in ["./", "src/", "../"]:
        path = possible_dirs + "internal_decoder.binary"
        if pathlib.Path(path).exists():
            return path
    return None


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

        path = internal_decoder_path()
        if path is None:
            raise RuntimeError(
                "You need an `internal_decoder.binary` file in the working directory to "
                "use `use_internal_decoder=True`.")

        command = (f"{path} "
                   f"-mode fi_match_from_dem "
                   f"-dem_fname '{dem_file}' "
                   f"-dets_fname '{dets_file}' "
                   f"-ignore_distance_1_errors "
                   f"-out '{out_file}'")
        if USE_CORRELATIONS:
            command += " -cheap_corr -edge_corr -node_corr"
        try:
            subprocess.check_output(command, shell=True)
        except:
            with open(dem_file) as f:
                with open("repro.dem", "w") as f2:
                    print(f.read(), file=f2)
            with open(dets_file) as f:
                with open("repro.dets", "w") as f2:
                    print(f.read(), file=f2)
            with open("repro.stim", "w") as f2:
                print(circuit, file=f2)
            print(f"Wrote case to `repro.dem`, `repro.dets`, and `repro.stim`.\nCommand line is: {command}", file=sys.stderr)
            raise

        predictions = np.zeros(shape=(num_shots, num_obs), dtype=np.bool8)
        with open(out_file, "r") as f:
            for shot in range(num_shots):
                for obs_index in range(num_obs):
                    c = f.read(1)
                    assert c in '01'
                    predictions[shot, obs_index] = c == '1'
                assert f.read(1) == '\n'

        return predictions


def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel) -> nx.Graph:
    """Convert a stim error model into a NetworkX graph."""
    det_offset = 0

    def _iter_model(m: stim.DetectorErrorModel, reps: int, callback: Callable[[float, List[int], List[int]], None]):
        nonlocal det_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _iter_model(instruction.body_copy(), instruction.repeat_count, callback)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: List[int] = []
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            elif t.is_separator():
                                # Treat each component of a decomposed error as an independent error.
                                # (Ideally we could configure some sort of correlated analysis; oh well.)
                                callback(p, dets, frames)
                                frames = []
                                dets = []
                        # Handle last component.
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

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets.append(num_detectors)
        if len(dets) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["qubit_id"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if set(old_frame_changes) == set(frame_changes):
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
        g.add_edge(*dets, weight=math.log((1 - p) / p), qubit_id=frame_changes, error_probability=p)

    _iter_model(model, 1, handle_error)

    return g


def detector_error_model_to_pymatching_graph(model: stim.DetectorErrorModel) -> pymatching.Matching:
    """Convert a stim error model into a pymatching graph."""
    g = detector_error_model_to_nx_graph(model)
    num_detectors = model.num_detectors
    num_observables = model.num_observables

    # Add spandrels to the graph to ensure pymatching will accept it.
    # - Make sure there's only one connected component.
    # - Make sure no detector nodes are skipped.
    # - Make sure no observable nodes are skipped.
    for k in range(num_detectors):
        g.add_node(k)
    g.add_node(num_detectors, is_boundary=True)
    g.add_node(num_detectors + 1)
    for k in range(num_detectors + 1):
        g.add_edge(k, num_detectors + 1, weight=9999999999)
    g.add_edge(num_detectors, num_detectors + 1, weight=9999999999, qubit_id=list(range(num_observables)))

    return pymatching.Matching(g)
