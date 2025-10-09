import re
import multiprocessing
from time import sleep
from typing import Callable, List, Any
from multiprocessing.pool import Pool, ApplyResult
from tqdm import tqdm
from prettytable import PrettyTable


def apply_args_and_kwargs(func:Callable, args, kwargs):
    return func(*args, **kwargs)


def starmap_async_with_kwargs(pool:Pool, func:Callable, *args, **kwargs):
    return pool.starmap_async(apply_args_and_kwargs, [(func, args, kwargs)])


def spawn_processes(func:Callable, datalist:List[Any], num_processes:int, verbose:bool, desc:str=None, *args, **kwargs):
    """
        For functions that the first argument is an element from `datalist` to handle in multi-process

        References: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/preprocessors/default_preprocessor.py#L232
        
        Note that don't pass a Python closure or a lambda expression here, as they're not pickle-able
    """
    if num_processes in [0, 1]:
        ret: List[Any] = []
        for data_item in tqdm(datalist, disable=verbose, desc=desc):
            ret.append(func(data_item, *args, **kwargs))
        return ret
    ret:List[ApplyResult] = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        remaining = list(range(len(datalist)))
        workers = [i for i in p._pool]
        for data_item in datalist:
            ret.append(
                starmap_async_with_kwargs(p, func, data_item, *args, **kwargs)
            )
        with tqdm(total=len(datalist), disable=verbose, desc=desc) as pbar:
            while len(remaining) > 0:
                all_alive = all([i.is_alive() for i in workers])
                if not all_alive: raise RuntimeError(f"One of the background processes is missing")
                finished = [i for i in remaining if ret[i].ready()]
                _ = [ret[i].get() for i in finished] # get done so that errors can be raised
                for _ in finished:
                    ret[_].get() # allows triggering errors
                    pbar.update()
                remaining = [i for i in remaining if i not in finished]
                sleep(0.1)
    return [i.get()[0] for i in ret]


def get_client_id(client_name: str):
    try:
        client_id = int(client_name.replace("client", ""))
    except:
        num_str = re.sub(r"\D", "", client_name)
        client_id = int(num_str) if len(num_str) > 0 else 0
    return client_id


def pretty_results(res: dict):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    for m in sorted(res.keys()):
        table.add_row([m, res[m]])
    return str(table)
