from __future__ import annotations
import os
# remove warnings
os.environ["nnUNet_raw"] = "./"
os.environ["nnUNet_preprocessed"] = "./"
os.environ["nnUNet_results"] = "./"

import argparse
import numpy as np
from typing import Dict, Tuple, List
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2_utils.experiment_planner import CustomizedExperimentPlanner
from nnunetv2_utils.default_preprocessor import CustomizedPreprocessor, nnUNetComponentUtils
from nnunetv2_utils.fingerprint_extractor import CustomizedDatasetFingerprintExtractor
from nnunetv2_utils.path import get_paths
from data.data_manager import CustomizedDataManager


def extract_fingerprint_dataset(dataset_id: int,
                                path_conf: Dict[str, str] | str,
                                num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    raw_folder, _, _ = get_paths(path_conf, dataset_id)
    dataset_name = os.path.basename(raw_folder)
    print(dataset_name, flush=True)

    if check_dataset_integrity:
        verify_dataset_integrity(raw_folder, num_processes)

    fpe = CustomizedDatasetFingerprintExtractor(dataset_id, path_conf, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)


def extract_fingerprints(dataset_ids: List[int],
                         path_conf: Dict[str, str] | str,
                         num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                         clean: bool = True, verbose: bool = True):


    for d in dataset_ids:
        extract_fingerprint_dataset(d, path_conf, num_processes, check_dataset_integrity, clean,
                                    verbose)


def plan_experiment_dataset(dataset_id: int,
                            path_conf: Dict[str, str] | str,
                            shape_must_be_divisible_by: np.ndarray,
                            gpu_memory_target_in_gb: float = None
                            ) -> Tuple[dict, str]:
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    kwargs = {}
    if gpu_memory_target_in_gb is not None:
        kwargs['gpu_memory_target_in_gb'] = gpu_memory_target_in_gb

    planner = CustomizedExperimentPlanner(dataset_id, path_conf=path_conf, **kwargs)
    ret = planner.plan_experiment(shape_must_be_divisible_by=shape_must_be_divisible_by)
    return ret, planner.plans_identifier


def plan_experiments(dataset_ids: List[int], 
                     path_conf: Dict[str, str] | str,
                     shape_must_be_divisible_by: np.ndarray,
                     gpu_memory_target_in_gb: float = None
                     ):
    plans_identifier = None
    for d in dataset_ids:
        _, plans_identifier = plan_experiment_dataset(d, path_conf, shape_must_be_divisible_by, gpu_memory_target_in_gb)
    return plans_identifier


def preprocess_dataset(dataset_id: int,
                       path_conf: Dict[str, str] | str,
                       plans_identifier: str = 'nnUNetPlansKDivisible',
                       configurations: Tuple[str] | List[str] = ('3d_fullres', '3d_lowres'),
                       num_processes: int | Tuple[int, ...] | List[int] = (8, 4, 8),
                       verbose: bool = False) -> None:
    if isinstance(num_processes, int):
        num_processes = [num_processes]
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    raw_folder, preprocessed_folder, _ = get_paths(path_conf, dataset_id)
    dataset_name = os.path.basename(raw_folder)
    print(f'Preprocessing dataset {dataset_name}', flush=True)
    plans_file = join(preprocessed_folder, plans_identifier + '.json')
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...', flush=True)
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.", flush=True)
            continue
        preprocessor = CustomizedPreprocessor(path_conf=path_conf, verbose=verbose)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)


    from distutils.file_util import copy_file
    maybe_mkdir_p(join(preprocessed_folder, 'gt_segmentations'))
    dataset_json = load_json(join(raw_folder, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(raw_folder, dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(preprocessed_folder, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)


def preprocess(dataset_ids: List[int],
               path_conf: Dict[str, str] | str,
               plans_identifier: str = 'nnUNetPlansKDivisible',
               configurations: Tuple[str] | List[str] = ('2d', '3d_fullres', '3d_lowres'),
               num_processes: int | Tuple[int, ...] | List[int] = (8, 4, 8),
               verbose: bool = False):
    for d in dataset_ids:
        preprocess_dataset(d, path_conf, plans_identifier, configurations, num_processes, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocessing Script")
    parser.add_argument("-c", default="src/path_conf.json", type=str, help="PATH configuration")
    parser.add_argument("-d", type=int, nargs="+", required=True, help="Dataset IDs")
    parser.add_argument("-gpu_memory_target_in_gb", type=float, default=8, help="Desired GPU memory when using AMP"),
    parser.add_argument("-np", type=int, default=6, help="number of processes")
    parser.add_argument("-no_pp", action="store_true", help="toggle planning only")
    parser.add_argument("-verbose", action="store_true", help="shows details")
    ARGS = parser.parse_args()

    print("Fingerprint extraction...", flush=True)
    extract_fingerprints(ARGS.d, ARGS.c, ARGS.np, True, True, True)

    plans_identifier = plan_experiments(ARGS.d, ARGS.c, [32, 32, 32], ARGS.gpu_memory_target_in_gb)

    if ARGS.np is None:
        default_np = {"3d_fullres": 4, "3d_lowres": 8}
        n_proc = [default_np[c] if c in default_np.keys() else 4 for c in ["3d_fullres"]]
    else:
        n_proc = ARGS.np

    if not ARGS.no_pp:
        print('Preprocessing...', flush=True)
        preprocess(ARGS.d, ARGS.c, plans_identifier, ["3d_fullres"], n_proc, ARGS.verbose)
        # unpack dataset...
        dataset_info = nnUNetComponentUtils(ARGS.d, ARGS.c)
        CustomizedDataManager(dataset_info, do_unpack_dataset=True)